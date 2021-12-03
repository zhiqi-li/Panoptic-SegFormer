import itertools
import logging
import os.path as osp
import tempfile
import warnings
from collections import OrderedDict
from easymd.datasets.panopticapi.converter_2cpng2pan import converter_memory

import mmcv
import numpy as np
from mmcv.utils import print_log
from terminaltables import AsciiTable

from mmdet.core import eval_recalls
from mmdet.datasets.api_wrappers import COCO, COCOeval
from mmdet.datasets.builder import DATASETS
from mmdet.datasets.custom import CustomDataset
from easymd.datasets.panopticapi import pq_compute,pq_compute2
from easymd.datasets.panopticapi import converter

@DATASETS.register_module()
class ADE_panoptic(CustomDataset):

    CLASSES = ['wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road, route', 'bed', 'window ', 
    'grass', 'cabinet', 'sidewalk, pavement', 'person', 'earth, ground', 'door', 'table', 'mountain,mount', 'plant', 'curtain', 'chair', 'car', 'water',
     'painting, picture', 'sofa', 'shelf', 'house',
    'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock, stone', 'wardrobe, closet, press',
    'lamp', 'tub', 'rail', 'cushion', 'base, pedestal, stand', 'box', 'column, pillar', 'signboard, sign', 
    'chest of drawers, chest, bureau, dresser', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator, icebox',
    'grandstand, covered stand', 'path', 'stairs', 'runway', 'case, display case, showcase, vitrine', 'pool table, billiard table, snooker table',
    'pillow', 'screen door, screen', 'stairway, staircase', 'river', 'bridge, span', 'bookcase', 'blind, screen',
    'coffee table', 'toilet, can, commode, crapper, pot, potty, stool, throne', 'flower', 'book', 'hill', 'bench', 'countertop', 
    'stove', 'palm, palm tree', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel, hut, hutch, shack, shanty',
    'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning, sunshade, sunblind', 'street lamp', 'booth', 'tv', 'plane', 
    'dirt track', 'clothes', 'pole', 'land, ground, soil', 'bannister, banister, balustrade, balusters, handrail', 'escalator, moving staircase, moving stairway',
    'ottoman, pouf, pouffe, puff, hassock', 'bottle', 'buffet, counter, sideboard', 'poster, posting, placard, notice, bill, card', 
    'stage', 'van', 'ship', 'fountain', 'conveyer belt, conveyor belt, conveyer, conveyor, transporter', 'canopy', 'washer, automatic washer, washing machine',
    'plaything, toy', 'pool', 'stool', 'barrel, cask', 'basket, handbasket', 'falls', 'tent', 'bag', 'minibike, motorbike', 'cradle', 'oven', 'ball', 
    'food, solid food', 'step, stair', 'tank, storage tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen',
    'blanket, cover', 'sculpture', 'hood, exhaust hood', 'sconce', 'vase', 'traffic light', 'tray', 'trash can', 'fan', 'pier', 'crt screen',
    'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass, drinking glass', 'clock', 'flag']

  
    def __init__(self,
           
            img_info_file_folder = './datasets/panoptic_val2017_detection_format.json',
            output_folder = 'seg',
            pred_json = 'pred.json',
            segmentations_folder='seg',
            gt_json = './datasets/ADEChallengeData2016/ade20k_panoptic_val.json',
            gt_folder = './datasets/ADEChallengeData2016/ade20k_panoptic_val',
            **kwarags):
            self.img_info_file_folder = img_info_file_folder
            self.output_folder =output_folder
            self.pred_json = pred_json
            self.gt_json = gt_json
            self.gt_folder =gt_folder
            self.segmentations_folder=segmentations_folder
            super(ADE_panoptic,self).__init__(**kwarags)
    def load_annotations(self, ann_file):
        """Load annotation from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation info from COCO api.
        """

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids()#(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}
        self.img_ids = self.coco.get_img_ids()
        data_infos = []
        total_ann_ids = []
        for i in self.img_ids:
            info = self.coco.load_imgs([i])[0]
            info['filename'] = info['file_name']
            data_infos.append(info)
            ann_ids = self.coco.get_ann_ids(img_ids=[i])
            total_ann_ids.extend(ann_ids)
        assert len(set(total_ann_ids)) == len(
            total_ann_ids), f"Annotation ids in '{ann_file}' are not unique!"
        return data_infos

    def get_ann_info(self, idx):
        """Get COCO annotation by index.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Annotation info of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return self._parse_ann_info(self.data_infos[idx], ann_info)

    def get_cat_ids(self, idx):
        """Get COCO category ids by index.

        Args:
            idx (int): Index of data.

        Returns:
            list[int]: All categories in the image of specified index.
        """

        img_id = self.data_infos[idx]['id']
        ann_ids = self.coco.get_ann_ids(img_ids=[img_id])
        ann_info = self.coco.load_anns(ann_ids)
        return [ann['category_id'] for ann in ann_info]

    def _filter_imgs(self, min_size=32):
        """Filter images too small or without ground truths."""
        valid_inds = []
        # obtain images that contain annotation
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        # obtain images that contain annotations of the required categories
        ids_in_cat = set()
        for i, class_id in enumerate(self.cat_ids):
            ids_in_cat |= set(self.coco.cat_img_map[class_id])
        # merge the image id sets of the two conditions and use the merged set
        # to filter out images if self.filter_empty_gt=True
        ids_in_cat &= ids_with_ann

        valid_img_ids = []
        for i, img_info in enumerate(self.data_infos):
            img_id = self.img_ids[i]
            if self.filter_empty_gt and img_id not in ids_in_cat:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
                valid_img_ids.append(img_id)
        self.img_ids = valid_img_ids
        return valid_inds

    def _parse_ann_info(self, img_info, ann_info):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,\
                labels, masks, seg_map. "masks" are raw annotations and not \
                decoded into binary masks.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        gt_masks_ann = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            inter_w = max(0, min(x1 + w, img_info['width']) - max(x1, 0))
            inter_h = max(0, min(y1 + h, img_info['height']) - max(y1, 0))
            if inter_w * inter_h == 0:
                continue
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            if ann['category_id'] not in self.cat_ids:
                continue
            bbox = [x1, y1, x1 + w, y1 + h]
            if ann.get('iscrowd', False):
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
                gt_masks_ann.append(ann.get('segmentation', None))

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        seg_map = img_info['filename'].replace('jpg', 'png')

        ann = dict(
            bboxes=gt_bboxes,
            labels=gt_labels,
            bboxes_ignore=gt_bboxes_ignore,
            masks=gt_masks_ann,
            seg_map=seg_map)

        return ann

    def xyxy2xywh(self, bbox):
        """Convert ``xyxy`` style bounding boxes to ``xywh`` style for COCO
        evaluation.

        Args:
            bbox (numpy.ndarray): The bounding boxes, shape (4, ), in
                ``xyxy`` order.

        Returns:
            list[float]: The converted bounding boxes, in ``xywh`` order.
        """

        _bbox = bbox.tolist()
        return [
            _bbox[0],
            _bbox[1],
            _bbox[2] - _bbox[0],
            _bbox[3] - _bbox[1],
        ]

    def _proposal2json(self, results):
        """Convert proposal results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            bboxes = results[idx]
            for i in range(bboxes.shape[0]):
                data = dict()
                data['image_id'] = img_id
                data['bbox'] = self.xyxy2xywh(bboxes[i])
                data['score'] = float(bboxes[i][4])
                data['category_id'] = 1
                json_results.append(data)
        return json_results

    def _det2json(self, results):
        """Convert detection results to COCO json style."""
        json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            result = results[idx]
            for label in range(len(result)):
                bboxes = result[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    json_results.append(data)
        return json_results

    def _segm2json(self, results):
        """Convert instance segmentation results to COCO json style."""
        bbox_json_results = []
        segm_json_results = []
        for idx in range(len(self)):
            img_id = self.img_ids[idx]
            det, seg = results[idx]
            for label in range(len(det)):
                # bbox results
                bboxes = det[label]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(bboxes[i][4])
                    data['category_id'] = self.cat_ids[label]
                    bbox_json_results.append(data)

                # segm results
                # some detectors use different scores for bbox and mask
                if isinstance(seg, tuple):
                    segms = seg[0][label]
                    mask_score = seg[1][label]
                else:
                    segms = seg[label]
                    mask_score = [bbox[4] for bbox in bboxes]
                for i in range(bboxes.shape[0]):
                    data = dict()
                    data['image_id'] = img_id
                    data['bbox'] = self.xyxy2xywh(bboxes[i])
                    data['score'] = float(mask_score[i])
                    data['category_id'] = self.cat_ids[label]
                    if isinstance(segms[i]['counts'], bytes):
                        segms[i]['counts'] = segms[i]['counts'].decode()
                    data['segmentation'] = segms[i]
                    segm_json_results.append(data)
        return bbox_json_results, segm_json_results

    def results2json(self, results, outfile_prefix):
        """Dump the detection results to a COCO style json file.

        There are 3 types of results: proposals, bbox predictions, mask
        predictions, and they have different data types. This method will
        automatically recognize the type, and dump them to json files.

        Args:
            results (list[list | tuple | ndarray]): Testing results of the
                dataset.
            outfile_prefix (str): The filename prefix of the json files. If the
                prefix is "somepath/xxx", the json files will be named
                "somepath/xxx.bbox.json", "somepath/xxx.segm.json",
                "somepath/xxx.proposal.json".

        Returns:
            dict[str: str]: Possible keys are "bbox", "segm", "proposal", and \
                values are corresponding filenames.
        """
        result_files = dict()
        if isinstance(results[0], list):
            json_results = self._det2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            mmcv.dump(json_results, result_files['bbox'])
        elif isinstance(results[0], tuple):
            json_results = self._segm2json(results)
            result_files['bbox'] = f'{outfile_prefix}.bbox.json'
            result_files['proposal'] = f'{outfile_prefix}.bbox.json'
            result_files['segm'] = f'{outfile_prefix}.segm.json'
            mmcv.dump(json_results[0], result_files['bbox'])
            mmcv.dump(json_results[1], result_files['segm'])
        elif isinstance(results[0], np.ndarray):
            json_results = self._proposal2json(results)
            result_files['proposal'] = f'{outfile_prefix}.proposal.json'
            mmcv.dump(json_results, result_files['proposal'])
        else:
            raise TypeError('invalid type of results')
        return result_files

    def fast_eval_recall(self, results, proposal_nums, iou_thrs, logger=None):
        gt_bboxes = []
        for i in range(len(self.img_ids)):
            ann_ids = self.coco.get_ann_ids(img_ids=self.img_ids[i])
            ann_info = self.coco.load_anns(ann_ids)
            if len(ann_info) == 0:
                gt_bboxes.append(np.zeros((0, 4)))
                continue
            bboxes = []
            for ann in ann_info:
                if ann.get('ignore', False) or ann['iscrowd']:
                    continue
                x1, y1, w, h = ann['bbox']
                bboxes.append([x1, y1, x1 + w, y1 + h])
            bboxes = np.array(bboxes, dtype=np.float32)
            if bboxes.shape[0] == 0:
                bboxes = np.zeros((0, 4))
            gt_bboxes.append(bboxes)

        recalls = eval_recalls(
            gt_bboxes, results, proposal_nums, iou_thrs, logger=logger)
        ar = recalls.mean(axis=1)
        return ar

    def format_results(self, results, jsonfile_prefix=None, **kwargs):
        """Format the results to json (standard format for COCO evaluation).

        Args:
            results (list[tuple | numpy.ndarray]): Testing results of the
                dataset.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.

        Returns:
            tuple: (result_files, tmp_dir), result_files is a dict containing \
                the json filepaths, tmp_dir is the temporal directory created \
                for saving json files when jsonfile_prefix is not specified.
        """
        assert isinstance(results, list), 'results must be a list'
        assert len(results) == len(self), (
            'The length of results is not equal to the dataset len: {} != {}'.
            format(len(results), len(self)))

        if jsonfile_prefix is None:
            tmp_dir = tempfile.TemporaryDirectory()
            jsonfile_prefix = osp.join(tmp_dir.name, 'results')
        else:
            tmp_dir = None
        result_files = self.results2json(results, jsonfile_prefix)
        return result_files, tmp_dir

    def evaluate(self,
                 results,
                 metric='bbox',
                 logger=None,
                 jsonfile_prefix=None,
                 classwise=False,
                 proposal_nums=(100, 300, 1000),
                 iou_thrs=None,
                 metric_items=None,
                 **kwargs):
        """Evaluation in COCO protocol.

        Args:
            results (list[list | tuple]): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated. Options are
                'bbox', 'segm', 'proposal', 'proposal_fast'.
            logger (logging.Logger | str | None): Logger used for printing
                related information during evaluation. Default: None.
            jsonfile_prefix (str | None): The prefix of json files. It includes
                the file path and the prefix of filename, e.g., "a/b/prefix".
                If not specified, a temp file will be created. Default: None.
            classwise (bool): Whether to evaluating the AP for each class.
            proposal_nums (Sequence[int]): Proposal number used for evaluating
                recalls, such as recall@100, recall@1000.
                Default: (100, 300, 1000).
            iou_thrs (Sequence[float], optional): IoU threshold used for
                evaluating recalls/mAPs. If set to a list, the average of all
                IoUs will also be computed. If not specified, [0.50, 0.55,
                0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95] will be used.
                Default: None.
            metric_items (list[str] | str, optional): Metric items that will
                be returned. If not specified, ``['AR@100', 'AR@300',
                'AR@1000', 'AR_s@1000', 'AR_m@1000', 'AR_l@1000' ]`` will be
                used when ``metric=='proposal'``, ``['mAP', 'mAP_50', 'mAP_75',
                'mAP_s', 'mAP_m', 'mAP_l']`` will be used when
                ``metric=='bbox' or metric=='segm'``.

        Returns:
            dict[str, float]: COCO style evaluation metric.
        """


        metrics = metric if isinstance(metric, list) else [metric]

        eval_results = OrderedDict()
        if 'panoptic' in metrics:
           
            assert 'panoptic' in results.keys()
            panoptic_result = results['panoptic']
            
            results_pq = pq_compute2(self.gt_json,panoptic_result, self.gt_folder, self.segmentations_folder,logger=logger)
            eval_results.update(results_pq) 
            metrics = [metric for metric in metrics if metric !='panoptic']
        
        if 'bbox' in results.keys() and 'segm' not in results.keys():
            results = results['bbox']
        elif  'bbox' in results.keys() and 'segm' in results.keys():
            results = [(bbox,segm) for bbox,segm in zip(results['bbox'],results['segm'])]


        allowed_metrics = ['bbox', 'segm', 'proposal', 'proposal_fast']
        for metric in metrics:
            if metric not in allowed_metrics:
                raise KeyError(f'metric {metric} is not supported')
        if iou_thrs is None:
            iou_thrs = np.linspace(
                .5, 0.95, int(np.round((0.95 - .5) / .05)) + 1, endpoint=True)
        if metric_items is not None:
            if not isinstance(metric_items, list):
                metric_items = [metric_items]
        
        result_files, tmp_dir = self.format_results(results, jsonfile_prefix)

        
        cocoGt = self.coco
        for metric in metrics:
            msg = f'Evaluating {metric}...'
            if logger is None:
                msg = '\n' + msg
            print_log(msg, logger=logger)

            if metric == 'proposal_fast':
                ar = self.fast_eval_recall(
                    results, proposal_nums, iou_thrs, logger='silent')
                log_msg = []
                for i, num in enumerate(proposal_nums):
                    eval_results[f'AR@{num}'] = ar[i]
                    log_msg.append(f'\nAR@{num}\t{ar[i]:.4f}')
                log_msg = ''.join(log_msg)
                print_log(log_msg, logger=logger)
                continue

            iou_type = 'bbox' if metric == 'proposal' else metric
            if metric not in result_files:
                raise KeyError(f'{metric} is not in results')
            try:
                predictions = mmcv.load(result_files[metric])
                if iou_type == 'segm':
                    # Refer to https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/coco.py#L331  # noqa
                    # When evaluating mask AP, if the results contain bbox,
                    # cocoapi will use the box area instead of the mask area
                    # for calculating the instance area. Though the overall AP
                    # is not affected, this leads to different
                    # small/medium/large mask AP results.
                    for x in predictions:
                        x.pop('bbox')
                    warnings.simplefilter('once')
                    warnings.warn(
                        'The key "bbox" is deleted for more accurate mask AP '
                        'of small/medium/large instances since v2.12.0. This '
                        'does not change the overall mAP calculation.',
                        UserWarning)
                cocoDt = cocoGt.loadRes(predictions)
            except IndexError:
                print_log(
                    'The testing results of the whole dataset is empty.',
                    logger=logger,
                    level=logging.ERROR)
                break

            cocoEval = COCOeval(cocoGt, cocoDt, iou_type)
            cocoEval.params.catIds = self.cat_ids
            cocoEval.params.imgIds = self.img_ids
            cocoEval.params.maxDets = list(proposal_nums)
            cocoEval.params.iouThrs = iou_thrs
            # mapping of cocoEval.stats
            coco_metric_names = {
                'mAP': 0,
                'mAP_50': 1,
                'mAP_75': 2,
                'mAP_s': 3,
                'mAP_m': 4,
                'mAP_l': 5,
                'AR@100': 6,
                'AR@300': 7,
                'AR@1000': 8,
                'AR_s@1000': 9,
                'AR_m@1000': 10,
                'AR_l@1000': 11
            }
            if metric_items is not None:
                for metric_item in metric_items:
                    if metric_item not in coco_metric_names:
                        raise KeyError(
                            f'metric item {metric_item} is not supported')

            if metric == 'proposal':
                cocoEval.params.useCats = 0
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if metric_items is None:
                    metric_items = [
                        'AR@100', 'AR@300', 'AR@1000', 'AR_s@1000',
                        'AR_m@1000', 'AR_l@1000'
                    ]

                for item in metric_items:
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[item]]:.3f}')
                    eval_results[item] = val
            else:
                cocoEval.evaluate()
                cocoEval.accumulate()
                cocoEval.summarize()
                if classwise:  # Compute per-category AP
                    # Compute per-category AP
                    # from https://github.com/facebookresearch/detectron2/
                    precisions = cocoEval.eval['precision']
                    # precision: (iou, recall, cls, area range, max dets)
                    assert len(self.cat_ids) == precisions.shape[2]

                    results_per_category = []
                    for idx, catId in enumerate(self.cat_ids):
                        # area range index 0: all area ranges
                        # max dets index -1: typically 100 per image
                        nm = self.coco.loadCats(catId)[0]
                        precision = precisions[:, :, idx, 0, -1]
                        precision = precision[precision > -1]
                        if precision.size:
                            ap = np.mean(precision)
                        else:
                            ap = float('nan')
                        results_per_category.append(
                            (f'{nm["name"]}', f'{float(ap):0.3f}'))

                    num_columns = min(6, len(results_per_category) * 2)
                    results_flatten = list(
                        itertools.chain(*results_per_category))
                    headers = ['category', 'AP'] * (num_columns // 2)
                    results_2d = itertools.zip_longest(*[
                        results_flatten[i::num_columns]
                        for i in range(num_columns)
                    ])
                    table_data = [headers]
                    table_data += [result for result in results_2d]
                    table = AsciiTable(table_data)
                    print_log('\n' + table.table, logger=logger)

                if metric_items is None:
                    metric_items = [
                        'mAP', 'mAP_50', 'mAP_75', 'mAP_s', 'mAP_m', 'mAP_l'
                    ]

                for metric_item in metric_items:
                    key = f'{metric}_{metric_item}'
                    val = float(
                        f'{cocoEval.stats[coco_metric_names[metric_item]]:.3f}'
                    )
                    eval_results[key] = val
                ap = cocoEval.stats[:6]
                eval_results[f'{metric}_mAP_copypaste'] = (
                    f'{ap[0]:.3f} {ap[1]:.3f} {ap[2]:.3f} {ap[3]:.3f} '
                    f'{ap[4]:.3f} {ap[5]:.3f}')
        if tmp_dir is not None:
            tmp_dir.cleanup()
        return eval_results
PAN =[
    {
        "name": "wall",
        "id": 0,
        "isthing": 0,
        "color": [
            120,
            120,
            120
        ]
    },
    {
        "name": "building",
        "id": 1,
        "isthing": 0,
        "color": [
            180,
            120,
            120
        ]
    },
    {
        "name": "sky",
        "id": 2,
        "isthing": 0,
        "color": [
            6,
            230,
            230
        ]
    },
    {
        "name": "floor",
        "id": 3,
        "isthing": 0,
        "color": [
            80,
            50,
            50
        ]
    },
    {
        "name": "tree",
        "id": 4,
        "isthing": 0,
        "color": [
            4,
            200,
            3
        ]
    },
    {
        "name": "ceiling",
        "id": 5,
        "isthing": 0,
        "color": [
            120,
            120,
            80
        ]
    },
    {
        "name": "road, route",
        "id": 6,
        "isthing": 0,
        "color": [
            140,
            140,
            140
        ]
    },
    {
        "name": "bed",
        "id": 7,
        "isthing": 1,
        "color": [
            204,
            5,
            255
        ]
    },
    {
        "name": "window ",
        "id": 8,
        "isthing": 1,
        "color": [
            230,
            230,
            230
        ]
    },
    {
        "name": "grass",
        "id": 9,
        "isthing": 0,
        "color": [
            4,
            250,
            7
        ]
    },
    {
        "name": "cabinet",
        "id": 10,
        "isthing": 1,
        "color": [
            224,
            5,
            255
        ]
    },
    {
        "name": "sidewalk, pavement",
        "id": 11,
        "isthing": 0,
        "color": [
            235,
            255,
            7
        ]
    },
    {
        "name": "person",
        "id": 12,
        "isthing": 1,
        "color": [
            150,
            5,
            61
        ]
    },
    {
        "name": "earth, ground",
        "id": 13,
        "isthing": 0,
        "color": [
            120,
            120,
            70
        ]
    },
    {
        "name": "door",
        "id": 14,
        "isthing": 1,
        "color": [
            8,
            255,
            51
        ]
    },
    {
        "name": "table",
        "id": 15,
        "isthing": 1,
        "color": [
            255,
            6,
            82
        ]
    },
    {
        "name": "mountain, mount",
        "id": 16,
        "isthing": 0,
        "color": [
            143,
            255,
            140
        ]
    },
    {
        "name": "plant",
        "id": 17,
        "isthing": 0,
        "color": [
            204,
            255,
            4
        ]
    },
    {
        "name": "curtain",
        "id": 18,
        "isthing": 1,
        "color": [
            255,
            51,
            7
        ]
    },
    {
        "name": "chair",
        "id": 19,
        "isthing": 1,
        "color": [
            204,
            70,
            3
        ]
    },
    {
        "name": "car",
        "id": 20,
        "isthing": 1,
        "color": [
            0,
            102,
            200
        ]
    },
    {
        "name": "water",
        "id": 21,
        "isthing": 0,
        "color": [
            61,
            230,
            250
        ]
    },
    {
        "name": "painting, picture",
        "id": 22,
        "isthing": 1,
        "color": [
            255,
            6,
            51
        ]
    },
    {
        "name": "sofa",
        "id": 23,
        "isthing": 1,
        "color": [
            11,
            102,
            255
        ]
    },
    {
        "name": "shelf",
        "id": 24,
        "isthing": 1,
        "color": [
            255,
            7,
            71
        ]
    },
    {
        "name": "house",
        "id": 25,
        "isthing": 0,
        "color": [
            255,
            9,
            224
        ]
    },
    {
        "name": "sea",
        "id": 26,
        "isthing": 0,
        "color": [
            9,
            7,
            230
        ]
    },
    {
        "name": "mirror",
        "id": 27,
        "isthing": 1,
        "color": [
            220,
            220,
            220
        ]
    },
    {
        "name": "rug",
        "id": 28,
        "isthing": 0,
        "color": [
            255,
            9,
            92
        ]
    },
    {
        "name": "field",
        "id": 29,
        "isthing": 0,
        "color": [
            112,
            9,
            255
        ]
    },
    {
        "name": "armchair",
        "id": 30,
        "isthing": 1,
        "color": [
            8,
            255,
            214
        ]
    },
    {
        "name": "seat",
        "id": 31,
        "isthing": 1,
        "color": [
            7,
            255,
            224
        ]
    },
    {
        "name": "fence",
        "id": 32,
        "isthing": 1,
        "color": [
            255,
            184,
            6
        ]
    },
    {
        "name": "desk",
        "id": 33,
        "isthing": 1,
        "color": [
            10,
            255,
            71
        ]
    },
    {
        "name": "rock, stone",
        "id": 34,
        "isthing": 0,
        "color": [
            255,
            41,
            10
        ]
    },
    {
        "name": "wardrobe, closet, press",
        "id": 35,
        "isthing": 1,
        "color": [
            7,
            255,
            255
        ]
    },
    {
        "name": "lamp",
        "id": 36,
        "isthing": 1,
        "color": [
            224,
            255,
            8
        ]
    },
    {
        "name": "tub",
        "id": 37,
        "isthing": 1,
        "color": [
            102,
            8,
            255
        ]
    },
    {
        "name": "rail",
        "id": 38,
        "isthing": 1,
        "color": [
            255,
            61,
            6
        ]
    },
    {
        "name": "cushion",
        "id": 39,
        "isthing": 1,
        "color": [
            255,
            194,
            7
        ]
    },
    {
        "name": "base, pedestal, stand",
        "id": 40,
        "isthing": 0,
        "color": [
            255,
            122,
            8
        ]
    },
    {
        "name": "box",
        "id": 41,
        "isthing": 1,
        "color": [
            0,
            255,
            20
        ]
    },
    {
        "name": "column, pillar",
        "id": 42,
        "isthing": 1,
        "color": [
            255,
            8,
            41
        ]
    },
    {
        "name": "signboard, sign",
        "id": 43,
        "isthing": 1,
        "color": [
            255,
            5,
            153
        ]
    },
    {
        "name": "chest of drawers, chest, bureau, dresser",
        "id": 44,
        "isthing": 1,
        "color": [
            6,
            51,
            255
        ]
    },
    {
        "name": "counter",
        "id": 45,
        "isthing": 1,
        "color": [
            235,
            12,
            255
        ]
    },
    {
        "name": "sand",
        "id": 46,
        "isthing": 0,
        "color": [
            160,
            150,
            20
        ]
    },
    {
        "name": "sink",
        "id": 47,
        "isthing": 1,
        "color": [
            0,
            163,
            255
        ]
    },
    {
        "name": "skyscraper",
        "id": 48,
        "isthing": 0,
        "color": [
            140,
            140,
            200
        ]
    },
    {
        "name": "fireplace",
        "id": 49,
        "isthing": 1,
        "color": [
            250,
            10,
            15
        ]
    },
    {
        "name": "refrigerator, icebox",
        "id": 50,
        "isthing": 1,
        "color": [
            20,
            255,
            0
        ]
    },
    {
        "name": "grandstand, covered stand",
        "id": 51,
        "isthing": 0,
        "color": [
            31,
            255,
            0
        ]
    },
    {
        "name": "path",
        "id": 52,
        "isthing": 0,
        "color": [
            255,
            31,
            0
        ]
    },
    {
        "name": "stairs",
        "id": 53,
        "isthing": 1,
        "color": [
            255,
            224,
            0
        ]
    },
    {
        "name": "runway",
        "id": 54,
        "isthing": 0,
        "color": [
            153,
            255,
            0
        ]
    },
    {
        "name": "case, display case, showcase, vitrine",
        "id": 55,
        "isthing": 1,
        "color": [
            0,
            0,
            255
        ]
    },
    {
        "name": "pool table, billiard table, snooker table",
        "id": 56,
        "isthing": 1,
        "color": [
            255,
            71,
            0
        ]
    },
    {
        "name": "pillow",
        "id": 57,
        "isthing": 1,
        "color": [
            0,
            235,
            255
        ]
    },
    {
        "name": "screen door, screen",
        "id": 58,
        "isthing": 1,
        "color": [
            0,
            173,
            255
        ]
    },
    {
        "name": "stairway, staircase",
        "id": 59,
        "isthing": 0,
        "color": [
            31,
            0,
            255
        ]
    },
    {
        "name": "river",
        "id": 60,
        "isthing": 0,
        "color": [
            11,
            200,
            200
        ]
    },
    {
        "name": "bridge, span",
        "id": 61,
        "isthing": 0,
        "color": [
            255,
            82,
            0
        ]
    },
    {
        "name": "bookcase",
        "id": 62,
        "isthing": 1,
        "color": [
            0,
            255,
            245
        ]
    },
    {
        "name": "blind, screen",
        "id": 63,
        "isthing": 0,
        "color": [
            0,
            61,
            255
        ]
    },
    {
        "name": "coffee table",
        "id": 64,
        "isthing": 1,
        "color": [
            0,
            255,
            112
        ]
    },
    {
        "name": "toilet, can, commode, crapper, pot, potty, stool, throne",
        "id": 65,
        "isthing": 1,
        "color": [
            0,
            255,
            133
        ]
    },
    {
        "name": "flower",
        "id": 66,
        "isthing": 1,
        "color": [
            255,
            0,
            0
        ]
    },
    {
        "name": "book",
        "id": 67,
        "isthing": 1,
        "color": [
            255,
            163,
            0
        ]
    },
    {
        "name": "hill",
        "id": 68,
        "isthing": 0,
        "color": [
            255,
            102,
            0
        ]
    },
    {
        "name": "bench",
        "id": 69,
        "isthing": 1,
        "color": [
            194,
            255,
            0
        ]
    },
    {
        "name": "countertop",
        "id": 70,
        "isthing": 1,
        "color": [
            0,
            143,
            255
        ]
    },
    {
        "name": "stove",
        "id": 71,
        "isthing": 1,
        "color": [
            51,
            255,
            0
        ]
    },
    {
        "name": "palm, palm tree",
        "id": 72,
        "isthing": 1,
        "color": [
            0,
            82,
            255
        ]
    },
    {
        "name": "kitchen island",
        "id": 73,
        "isthing": 1,
        "color": [
            0,
            255,
            41
        ]
    },
    {
        "name": "computer",
        "id": 74,
        "isthing": 1,
        "color": [
            0,
            255,
            173
        ]
    },
    {
        "name": "swivel chair",
        "id": 75,
        "isthing": 1,
        "color": [
            10,
            0,
            255
        ]
    },
    {
        "name": "boat",
        "id": 76,
        "isthing": 1,
        "color": [
            173,
            255,
            0
        ]
    },
    {
        "name": "bar",
        "id": 77,
        "isthing": 0,
        "color": [
            0,
            255,
            153
        ]
    },
    {
        "name": "arcade machine",
        "id": 78,
        "isthing": 1,
        "color": [
            255,
            92,
            0
        ]
    },
    {
        "name": "hovel, hut, hutch, shack, shanty",
        "id": 79,
        "isthing": 0,
        "color": [
            255,
            0,
            255
        ]
    },
    {
        "name": "bus",
        "id": 80,
        "isthing": 1,
        "color": [
            255,
            0,
            245
        ]
    },
    {
        "name": "towel",
        "id": 81,
        "isthing": 1,
        "color": [
            255,
            0,
            102
        ]
    },
    {
        "name": "light",
        "id": 82,
        "isthing": 1,
        "color": [
            255,
            173,
            0
        ]
    },
    {
        "name": "truck",
        "id": 83,
        "isthing": 1,
        "color": [
            255,
            0,
            20
        ]
    },
    {
        "name": "tower",
        "id": 84,
        "isthing": 0,
        "color": [
            255,
            184,
            184
        ]
    },
    {
        "name": "chandelier",
        "id": 85,
        "isthing": 1,
        "color": [
            0,
            31,
            255
        ]
    },
    {
        "name": "awning, sunshade, sunblind",
        "id": 86,
        "isthing": 1,
        "color": [
            0,
            255,
            61
        ]
    },
    {
        "name": "street lamp",
        "id": 87,
        "isthing": 1,
        "color": [
            0,
            71,
            255
        ]
    },
    {
        "name": "booth",
        "id": 88,
        "isthing": 1,
        "color": [
            255,
            0,
            204
        ]
    },
    {
        "name": "tv",
        "id": 89,
        "isthing": 1,
        "color": [
            0,
            255,
            194
        ]
    },
    {
        "name": "plane",
        "id": 90,
        "isthing": 1,
        "color": [
            0,
            255,
            82
        ]
    },
    {
        "name": "dirt track",
        "id": 91,
        "isthing": 0,
        "color": [
            0,
            10,
            255
        ]
    },
    {
        "name": "clothes",
        "id": 92,
        "isthing": 1,
        "color": [
            0,
            112,
            255
        ]
    },
    {
        "name": "pole",
        "id": 93,
        "isthing": 1,
        "color": [
            51,
            0,
            255
        ]
    },
    {
        "name": "land, ground, soil",
        "id": 94,
        "isthing": 0,
        "color": [
            0,
            194,
            255
        ]
    },
    {
        "name": "bannister, banister, balustrade, balusters, handrail",
        "id": 95,
        "isthing": 1,
        "color": [
            0,
            122,
            255
        ]
    },
    {
        "name": "escalator, moving staircase, moving stairway",
        "id": 96,
        "isthing": 0,
        "color": [
            0,
            255,
            163
        ]
    },
    {
        "name": "ottoman, pouf, pouffe, puff, hassock",
        "id": 97,
        "isthing": 1,
        "color": [
            255,
            153,
            0
        ]
    },
    {
        "name": "bottle",
        "id": 98,
        "isthing": 1,
        "color": [
            0,
            255,
            10
        ]
    },
    {
        "name": "buffet, counter, sideboard",
        "id": 99,
        "isthing": 0,
        "color": [
            255,
            112,
            0
        ]
    },
    {
        "name": "poster, posting, placard, notice, bill, card",
        "id": 100,
        "isthing": 0,
        "color": [
            143,
            255,
            0
        ]
    },
    {
        "name": "stage",
        "id": 101,
        "isthing": 0,
        "color": [
            82,
            0,
            255
        ]
    },
    {
        "name": "van",
        "id": 102,
        "isthing": 1,
        "color": [
            163,
            255,
            0
        ]
    },
    {
        "name": "ship",
        "id": 103,
        "isthing": 1,
        "color": [
            255,
            235,
            0
        ]
    },
    {
        "name": "fountain",
        "id": 104,
        "isthing": 1,
        "color": [
            8,
            184,
            170
        ]
    },
    {
        "name": "conveyer belt, conveyor belt, conveyer, conveyor, transporter",
        "id": 105,
        "isthing": 0,
        "color": [
            133,
            0,
            255
        ]
    },
    {
        "name": "canopy",
        "id": 106,
        "isthing": 0,
        "color": [
            0,
            255,
            92
        ]
    },
    {
        "name": "washer, automatic washer, washing machine",
        "id": 107,
        "isthing": 1,
        "color": [
            184,
            0,
            255
        ]
    },
    {
        "name": "plaything, toy",
        "id": 108,
        "isthing": 1,
        "color": [
            255,
            0,
            31
        ]
    },
    {
        "name": "pool",
        "id": 109,
        "isthing": 0,
        "color": [
            0,
            184,
            255
        ]
    },
    {
        "name": "stool",
        "id": 110,
        "isthing": 1,
        "color": [
            0,
            214,
            255
        ]
    },
    {
        "name": "barrel, cask",
        "id": 111,
        "isthing": 1,
        "color": [
            255,
            0,
            112
        ]
    },
    {
        "name": "basket, handbasket",
        "id": 112,
        "isthing": 1,
        "color": [
            92,
            255,
            0
        ]
    },
    {
        "name": "falls",
        "id": 113,
        "isthing": 0,
        "color": [
            0,
            224,
            255
        ]
    },
    {
        "name": "tent",
        "id": 114,
        "isthing": 0,
        "color": [
            112,
            224,
            255
        ]
    },
    {
        "name": "bag",
        "id": 115,
        "isthing": 1,
        "color": [
            70,
            184,
            160
        ]
    },
    {
        "name": "minibike, motorbike",
        "id": 116,
        "isthing": 1,
        "color": [
            163,
            0,
            255
        ]
    },
    {
        "name": "cradle",
        "id": 117,
        "isthing": 0,
        "color": [
            153,
            0,
            255
        ]
    },
    {
        "name": "oven",
        "id": 118,
        "isthing": 1,
        "color": [
            71,
            255,
            0
        ]
    },
    {
        "name": "ball",
        "id": 119,
        "isthing": 1,
        "color": [
            255,
            0,
            163
        ]
    },
    {
        "name": "food, solid food",
        "id": 120,
        "isthing": 1,
        "color": [
            255,
            204,
            0
        ]
    },
    {
        "name": "step, stair",
        "id": 121,
        "isthing": 1,
        "color": [
            255,
            0,
            143
        ]
    },
    {
        "name": "tank, storage tank",
        "id": 122,
        "isthing": 0,
        "color": [
            0,
            255,
            235
        ]
    },
    {
        "name": "trade name",
        "id": 123,
        "isthing": 1,
        "color": [
            133,
            255,
            0
        ]
    },
    {
        "name": "microwave",
        "id": 124,
        "isthing": 1,
        "color": [
            255,
            0,
            235
        ]
    },
    {
        "name": "pot",
        "id": 125,
        "isthing": 1,
        "color": [
            245,
            0,
            255
        ]
    },
    {
        "name": "animal",
        "id": 126,
        "isthing": 1,
        "color": [
            255,
            0,
            122
        ]
    },
    {
        "name": "bicycle",
        "id": 127,
        "isthing": 1,
        "color": [
            255,
            245,
            0
        ]
    },
    {
        "name": "lake",
        "id": 128,
        "isthing": 0,
        "color": [
            10,
            190,
            212
        ]
    },
    {
        "name": "dishwasher",
        "id": 129,
        "isthing": 1,
        "color": [
            214,
            255,
            0
        ]
    },
    {
        "name": "screen",
        "id": 130,
        "isthing": 1,
        "color": [
            0,
            204,
            255
        ]
    },
    {
        "name": "blanket, cover",
        "id": 131,
        "isthing": 0,
        "color": [
            20,
            0,
            255
        ]
    },
    {
        "name": "sculpture",
        "id": 132,
        "isthing": 1,
        "color": [
            255,
            255,
            0
        ]
    },
    {
        "name": "hood, exhaust hood",
        "id": 133,
        "isthing": 1,
        "color": [
            0,
            153,
            255
        ]
    },
    {
        "name": "sconce",
        "id": 134,
        "isthing": 1,
        "color": [
            0,
            41,
            255
        ]
    },
    {
        "name": "vase",
        "id": 135,
        "isthing": 1,
        "color": [
            0,
            255,
            204
        ]
    },
    {
        "name": "traffic light",
        "id": 136,
        "isthing": 1,
        "color": [
            41,
            0,
            255
        ]
    },
    {
        "name": "tray",
        "id": 137,
        "isthing": 1,
        "color": [
            41,
            255,
            0
        ]
    },
    {
        "name": "trash can",
        "id": 138,
        "isthing": 1,
        "color": [
            173,
            0,
            255
        ]
    },
    {
        "name": "fan",
        "id": 139,
        "isthing": 1,
        "color": [
            0,
            245,
            255
        ]
    },
    {
        "name": "pier",
        "id": 140,
        "isthing": 0,
        "color": [
            71,
            0,
            255
        ]
    },
    {
        "name": "crt screen",
        "id": 141,
        "isthing": 0,
        "color": [
            122,
            0,
            255
        ]
    },
    {
        "name": "plate",
        "id": 142,
        "isthing": 1,
        "color": [
            0,
            255,
            184
        ]
    },
    {
        "name": "monitor",
        "id": 143,
        "isthing": 1,
        "color": [
            0,
            92,
            255
        ]
    },
    {
        "name": "bulletin board",
        "id": 144,
        "isthing": 1,
        "color": [
            184,
            255,
            0
        ]
    },
    {
        "name": "shower",
        "id": 145,
        "isthing": 0,
        "color": [
            0,
            133,
            255
        ]
    },
    {
        "name": "radiator",
        "id": 146,
        "isthing": 1,
        "color": [
            255,
            214,
            0
        ]
    },
    {
        "name": "glass, drinking glass",
        "id": 147,
        "isthing": 1,
        "color": [
            25,
            194,
            194
        ]
    },
    {
        "name": "clock",
        "id": 148,
        "isthing": 1,
        "color": [
            102,
            255,
            0
        ]
    },
    {
        "name": "flag",
        "id": 149,
        "isthing": 1,
        "color": [
            92,
            0,
            255
        ]
    }
]