#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json
import time
from datetime import timedelta
from collections import defaultdict
import argparse
import multiprocessing
from mmcv.utils import print_log
import PIL.Image as Image

from .utils import get_traceback, id2rgb, rgb2id, mask_to_boundary



OFFSET = 256 * 256 * 256
#VOID = 0

class PQStatCat():
        def __init__(self):
            self.iou = 0.0
            self.tp = 0
            self.fp = 0
            self.fn = 0

        def __iadd__(self, pq_stat_cat):
            self.iou += pq_stat_cat.iou
            self.tp += pq_stat_cat.tp
            self.fp += pq_stat_cat.fp
            self.fn += pq_stat_cat.fn
            return self


class PQStat():
    def __init__(self):
        self.pq_per_cat = defaultdict(PQStatCat)

    def __getitem__(self, i):
        return self.pq_per_cat[i]

    def __iadd__(self, pq_stat):
        for label, pq_stat_cat in pq_stat.pq_per_cat.items():
            self.pq_per_cat[label] += pq_stat_cat
        return self

    def pq_average(self, categories, isthing):
        pq, sq, rq, n = 0, 0, 0, 0
        per_class_results = {}
       
        for label, label_info in categories.items():
            if isthing is not None:
                cat_isthing = label_info['isthing'] == 1
                if isthing != cat_isthing:
                    continue
            iou = self.pq_per_cat[label].iou
            tp = self.pq_per_cat[label].tp
            fp = self.pq_per_cat[label].fp
            fn = self.pq_per_cat[label].fn
            
            if tp + fp + fn == 0:
                per_class_results[label] = {'pq': 0.0, 'sq': 0.0, 'rq': 0.0}
                continue
            n += 1
            pq_class = iou / (tp + 0.5 * fp + 0.5 * fn)
            sq_class = iou / tp if tp != 0 else 0
            rq_class = tp / (tp + 0.5 * fp + 0.5 * fn)
            per_class_results[label] = {'pq': pq_class, 'sq': sq_class, 'rq': rq_class}
            pq += pq_class
            sq += sq_class
            rq += rq_class
        pq = pq/n
        sq = sq/n
        rq = rq/n
        pq = float(f'{pq:0.3f}')
        sq = float(f'{sq:0.3f}')
        rq = float(f'{rq:0.3f}')
        return {'pq': pq, 'sq': sq, 'rq': rq , 'n': n}, per_class_results


@get_traceback
def pq_compute_single_core(proc_id, annotation_set, gt_folder, pred_folder, categories, iou_type, dilation_ratio,VOID=0):
    pq_stat = PQStat()
    if iou_type == "boundary":
        largest_cat_id = np.amax(list(categories.keys()))
        BOUNDARY_ID = largest_cat_id + 1
    else:
        BOUNDARY_ID = None

    idx = 0
    for gt_ann, pred_ann in annotation_set:
        #if idx % 100 == 0:
            #print('Core: {}, {} from {} images processed'.format(proc_id, idx, len(annotation_set)))
        # += 1

        pan_gt = np.array(Image.open(os.path.join(gt_folder, gt_ann['file_name'])), dtype=np.uint32)
        pan_gt = rgb2id(pan_gt)
        pan_pred = np.array(Image.open(os.path.join(pred_folder, pred_ann['file_name'])), dtype=np.uint32)
      
        pan_pred = rgb2id(pan_pred)
     
        gt_segms = {el['id']: el for el in gt_ann['segments_info']}
        pred_segms = {el['id']: el for el in pred_ann['segments_info']}

        # predicted segments area calculation + prediction sanity checks
        
        pred_labels_set = set(el['id'] for el in pred_ann['segments_info'])
        labels, labels_cnt = np.unique(pan_pred, return_counts=True)

        
        for label, label_cnt in zip(labels, labels_cnt):
            
            if label not in pred_segms:
                
                if label == VOID:
                    continue
                #print(colors)
                #print('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
                #print(label,pred_ann)
                raise KeyError('In the image with ID {} segment with ID {} is presented in PNG and not presented in JSON.'.format(gt_ann['image_id'], label))
            pred_segms[label]['area'] = label_cnt
            pred_labels_set.remove(label)
            if pred_segms[label]['category_id'] not in categories:
                #print('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
                raise KeyError('In the image with ID {} segment with ID {} has unknown category_id {}.'.format(gt_ann['image_id'], label, pred_segms[label]['category_id']))
        if len(pred_labels_set) != 0:
            #print('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))
            raise KeyError('In the image with ID {} the following segment IDs {} are presented in JSON and not presented in PNG.'.format(gt_ann['image_id'], list(pred_labels_set)))
        if iou_type == "boundary":
            pan_gt_boundary = pan_gt.copy()
            new_segments_info = []
            for el in gt_ann['segments_info']:
                binary_mask = (pan_gt_boundary == el['id']).astype(np.uint8)
                binary_boundary = mask_to_boundary(binary_mask, dilation_ratio)
                pan_gt_boundary[binary_mask > 0] = BOUNDARY_ID
                pan_gt_boundary[binary_boundary > 0] = el['id']
                # to calculate union
                el['boundary_area'] = np.sum(binary_boundary > 0)
                new_segments_info.append(el)
            gt_ann['segments_info'] = new_segments_info

            pan_pred_boundary = pan_pred.copy()
            new_segments_info = []
            for el in pred_ann['segments_info']:
                binary_mask = (pan_pred_boundary == el['id']).astype(np.uint8)
                binary_boundary = mask_to_boundary(binary_mask, dilation_ratio)
                pan_pred_boundary[binary_mask > 0] = BOUNDARY_ID
                pan_pred_boundary[binary_boundary > 0] = el['id']
                # to calculate union
                el['boundary_area'] = np.sum(binary_boundary > 0)
                new_segments_info.append(el)
            pred_ann['segments_info'] = new_segments_info

            # update
            gt_segms = {el['id']: el for el in gt_ann['segments_info']}
            pred_segms = {el['id']: el for el in pred_ann['segments_info']}
        else:
            pan_gt_boundary = None
            pan_pred_boundary = None

        # confusion matrix calculation
        pan_gt_pred = pan_gt.astype(np.uint64) * OFFSET + pan_pred.astype(np.uint64)
        gt_pred_map = {}
        labels, labels_cnt = np.unique(pan_gt_pred, return_counts=True)
        for label, intersection in zip(labels, labels_cnt):
            gt_id = label // OFFSET
            pred_id = label % OFFSET
            gt_pred_map[(gt_id, pred_id)] = intersection
        if iou_type == "boundary":
            pan_gt_pred_boundary = pan_gt_boundary.astype(np.uint64) * OFFSET + pan_pred_boundary.astype(np.uint64)
            gt_pred_map_boundary = {}
            labels, labels_cnt = np.unique(pan_gt_pred_boundary, return_counts=True)
            for label, intersection in zip(labels, labels_cnt):
                gt_id = label // OFFSET
                pred_id = label % OFFSET
                gt_pred_map_boundary[(gt_id, pred_id)] = intersection
        else:
            gt_pred_map_boundary = None

        # count all matched pairs
        gt_matched = set()
        pred_matched = set()
        for label_tuple, intersection in gt_pred_map.items():
            gt_label, pred_label = label_tuple
            if gt_label not in gt_segms:
                continue
            if pred_label not in pred_segms:
                continue
            if gt_segms[gt_label]['iscrowd'] == 1:
                continue
            if gt_segms[gt_label]['category_id'] != pred_segms[pred_label]['category_id']:
                continue

            union = pred_segms[pred_label]['area'] + gt_segms[gt_label]['area'] - intersection - gt_pred_map.get((VOID, pred_label), 0)
            iou = intersection / union
            if iou_type == "boundary":
                # 2 conditions
                # (1) if (gt_label, pred_label) pair does not exist in gt_pred_map_boundary, it means boundary intersection is 0
                # (2) no need to take care of (gt_label, pred_label) that is in gt_pred_map_boundary but not in gt_pred_map
                # because it will be ignored since this pair has mask iou = 0
                boundary_intersection = gt_pred_map_boundary.get(label_tuple, 0)
                boundary_union = (
                    pred_segms[pred_label]['boundary_area'] + gt_segms[gt_label]['boundary_area'] -
                    boundary_intersection - gt_pred_map_boundary.get((VOID, pred_label), 0)
                )
                boundary_iou = boundary_intersection / boundary_union
                # update iou with min(mask iou, boundary iou)
                iou = min(iou, boundary_iou)

            if iou > 0.5:
                pq_stat[gt_segms[gt_label]['category_id']].tp += 1
                pq_stat[gt_segms[gt_label]['category_id']].iou += iou
                gt_matched.add(gt_label)
                pred_matched.add(pred_label)

        # count false positives
        crowd_labels_dict = {}
        for gt_label, gt_info in gt_segms.items():
            if gt_label in gt_matched:
                continue
            # crowd segments are ignored
            if gt_info['iscrowd'] == 1:
                crowd_labels_dict[gt_info['category_id']] = gt_label
                continue
            pq_stat[gt_info['category_id']].fn += 1

        # count false positives
        for pred_label, pred_info in pred_segms.items():
            if pred_label in pred_matched:
                continue
            # intersection of the segment with VOID
            intersection = gt_pred_map.get((VOID, pred_label), 0)
            # plus intersection with corresponding CROWD region if it exists
            if pred_info['category_id'] in crowd_labels_dict:
                intersection += gt_pred_map.get((crowd_labels_dict[pred_info['category_id']], pred_label), 0)
            # predicted segment is ignored if more than half of the segment correspond to VOID and CROWD regions
            if intersection / pred_info['area'] > 0.5:
                continue
            pq_stat[pred_info['category_id']].fp += 1
        
    #print('Core: {}, all {} images processed'.format(proc_id, len(annotation_set)))
    return pq_stat


def pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories,iou_type, dilation_ratio,VOID):
    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(matched_annotations_list, cpu_num)
    #print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotation_set in enumerate(annotations_split):
        p = workers.apply_async(pq_compute_single_core,
                                (proc_id, annotation_set, gt_folder, pred_folder, categories, iou_type, dilation_ratio,VOID))
        processes.append(p)
    pq_stat = PQStat()
    for p in processes:
        pq_stat += p.get()
    return pq_stat



def pq_compute2(gt_json_file, pred_json, gt_folder=None, pred_folder=None,logger=None,iou_type="segm", dilation_ratio=0.02,VOID=0):


    start_time = time.time()
    if 'ade20k' in gt_json_file:
        VOID=151
    with open(gt_json_file, 'r') as f:
        gt_json = json.load(f)
    #with open(pred_json_file, 'r') as f:
    #    pred_json = json.load(f)
    if gt_folder is None:
        gt_folder = gt_json_file.replace('.json', '')
    if pred_folder is None:
        assert False
        #pred_folder = pred_json_file.replace('.json', '')
    categories = {el['id']: el for el in gt_json['categories']}

    print_log("Evaluation panoptic segmentation metrics:",logger=logger)
    print_log("Ground truth:",logger=logger)
    print_log("\tSegmentation folder: {}".format(gt_folder),logger=logger)
    print_log("\tJSON file: {}".format(gt_json_file),logger=logger)
    print_log("Prediction:",logger=logger)
    print_log("\tSegmentation folder: {}".format(pred_folder),logger=logger)
    #print_log("\tJSON file: {}".format(pred_json_file),logger=logger)
    
    if not os.path.isdir(gt_folder):
        raise Exception("Folder {} with ground truth segmentations doesn't exist".format(gt_folder))
    if not os.path.isdir(pred_folder):
        raise Exception("Folder {} with predicted segmentations doesn't exist".format(pred_folder))

    pred_annotations = {el['image_id']: el for el in pred_json}
    matched_annotations_list = []
    for gt_ann in gt_json['annotations']:
        image_id = gt_ann['image_id']
        if image_id not in pred_annotations:
            #print_log("\t {image_id} not in pred_annotations but in gt_annotations".format(image_id=image_id),logger=logger)
            continue
        
            #raise Exception('no prediction for the image with id: {}'.format(image_id))
        
        matched_annotations_list.append((gt_ann, pred_annotations[image_id]))

    pq_stat = pq_compute_multi_core(matched_annotations_list, gt_folder, pred_folder, categories, iou_type, dilation_ratio,VOID=VOID)

    metrics = [("All", None), ("Things", True), ("Stuff", False)]
    results = {}
    for name, isthing in metrics:
        results[name], per_class_results = pq_stat.pq_average(categories, isthing=isthing)
        if name == 'All':
            results['per_class'] = per_class_results
    print_log("{:10s}| {:>5s}  {:>5s}  {:>5s} {:>5s}".format("", "PQ", "SQ", "RQ", "N"),logger=logger)
    print_log("-" * (10 + 7 * 4),logger=logger)
    #print_log(results['per_class'],logger=logger)
    for name, _isthing in metrics:
        print_log("{:10s}| {:5.3f}  {:5.3f}  {:5.3f} {:5d}".format(
            name,
            100 * results[name]['pq'],
            100 * results[name]['sq'],
            100 * results[name]['rq'],
            results[name]['n']),logger=logger
        )

    t_delta = time.time() - start_time
    print_log("Time elapsed: {:0.2f} seconds".format(t_delta),logger=logger)
    return results


