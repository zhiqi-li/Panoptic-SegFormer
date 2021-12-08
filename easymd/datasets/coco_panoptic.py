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
from easymd.datasets.panopticapi import pq_compute2
from easymd.datasets.panopticapi import converter

@DATASETS.register_module()
class CocoDataset_panoptic(CustomDataset):

    CLASSES = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
        'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 
        'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
        'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
        'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 
        'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
        'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
        'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 
        'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
        'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush',
        'banner', 'blanket', 'bridge', 'cardboard', 'counter', 'curtain', 
        'door-stuff', 'floor-wood', 'flower', 'fruit', 'gravel', 'house',
        'light', 'mirror-stuff', 'net', 'pillow', 'platform','playingfield', 
        'railroad', 'river', 'road', 'roof', 'sand', 'sea', 'shelf', 'snow',
        'stairs', 'tent', 'towel', 'wall-brick', 'wall-stone', 'wall-tile', 
        'wall-wood', 'water-other', 'window-blind', 'window-other', 'tree-merged',
        'fence-merged', 'ceiling-merged', 'sky-other-merged', 'cabinet-merged', 
        'table-merged', 'floor-other-merged', 'pavement-merged', 'mountain-merged',
        'grass-merged', 'dirt-merged', 'paper-merged', 'food-other-merged',
        'building-other-merged', 'rock-merged', 'wall-other-merged', 'rug-merged']
  
    def __init__(self,
            segmentations_folder='seg',
            gt_json = './datasets/annotations/panoptic_val2017.json',
            gt_folder = './datasets/annotations/panoptic_val2017',
            **kwarags):
            self.gt_json = gt_json
            self.gt_folder =gt_folder
            self.segmentations_folder=segmentations_folder
            super(CocoDataset_panoptic,self).__init__(**kwarags)
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
id_and_category_maps =[
    {
        "supercategory": "person",
        "color": [
            220,
            20,
            60
        ],
        "isthing": 1,
        "id": 1,
        "name": "person"
    },
    {
        "supercategory": "vehicle",
        "color": [
            119,
            11,
            32
        ],
        "isthing": 1,
        "id": 2,
        "name": "bicycle"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            0,
            142
        ],
        "isthing": 1,
        "id": 3,
        "name": "car"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            0,
            230
        ],
        "isthing": 1,
        "id": 4,
        "name": "motorcycle"
    },
    {
        "supercategory": "vehicle",
        "color": [
            106,
            0,
            228
        ],
        "isthing": 1,
        "id": 5,
        "name": "airplane"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            60,
            100
        ],
        "isthing": 1,
        "id": 6,
        "name": "bus"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            80,
            100
        ],
        "isthing": 1,
        "id": 7,
        "name": "train"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            0,
            70
        ],
        "isthing": 1,
        "id": 8,
        "name": "truck"
    },
    {
        "supercategory": "vehicle",
        "color": [
            0,
            0,
            192
        ],
        "isthing": 1,
        "id": 9,
        "name": "boat"
    },
    {
        "supercategory": "outdoor",
        "color": [
            250,
            170,
            30
        ],
        "isthing": 1,
        "id": 10,
        "name": "traffic light"
    },
    {
        "supercategory": "outdoor",
        "color": [
            100,
            170,
            30
        ],
        "isthing": 1,
        "id": 11,
        "name": "fire hydrant"
    },
    {
        "supercategory": "outdoor",
        "color": [
            220,
            220,
            0
        ],
        "isthing": 1,
        "id": 13,
        "name": "stop sign"
    },
    {
        "supercategory": "outdoor",
        "color": [
            175,
            116,
            175
        ],
        "isthing": 1,
        "id": 14,
        "name": "parking meter"
    },
    {
        "supercategory": "outdoor",
        "color": [
            250,
            0,
            30
        ],
        "isthing": 1,
        "id": 15,
        "name": "bench"
    },
    {
        "supercategory": "animal",
        "color": [
            165,
            42,
            42
        ],
        "isthing": 1,
        "id": 16,
        "name": "bird"
    },
    {
        "supercategory": "animal",
        "color": [
            255,
            77,
            255
        ],
        "isthing": 1,
        "id": 17,
        "name": "cat"
    },
    {
        "supercategory": "animal",
        "color": [
            0,
            226,
            252
        ],
        "isthing": 1,
        "id": 18,
        "name": "dog"
    },
    {
        "supercategory": "animal",
        "color": [
            182,
            182,
            255
        ],
        "isthing": 1,
        "id": 19,
        "name": "horse"
    },
    {
        "supercategory": "animal",
        "color": [
            0,
            82,
            0
        ],
        "isthing": 1,
        "id": 20,
        "name": "sheep"
    },
    {
        "supercategory": "animal",
        "color": [
            120,
            166,
            157
        ],
        "isthing": 1,
        "id": 21,
        "name": "cow"
    },
    {
        "supercategory": "animal",
        "color": [
            110,
            76,
            0
        ],
        "isthing": 1,
        "id": 22,
        "name": "elephant"
    },
    {
        "supercategory": "animal",
        "color": [
            174,
            57,
            255
        ],
        "isthing": 1,
        "id": 23,
        "name": "bear"
    },
    {
        "supercategory": "animal",
        "color": [
            199,
            100,
            0
        ],
        "isthing": 1,
        "id": 24,
        "name": "zebra"
    },
    {
        "supercategory": "animal",
        "color": [
            72,
            0,
            118
        ],
        "isthing": 1,
        "id": 25,
        "name": "giraffe"
    },
    {
        "supercategory": "accessory",
        "color": [
            255,
            179,
            240
        ],
        "isthing": 1,
        "id": 27,
        "name": "backpack"
    },
    {
        "supercategory": "accessory",
        "color": [
            0,
            125,
            92
        ],
        "isthing": 1,
        "id": 28,
        "name": "umbrella"
    },
    {
        "supercategory": "accessory",
        "color": [
            209,
            0,
            151
        ],
        "isthing": 1,
        "id": 31,
        "name": "handbag"
    },
    {
        "supercategory": "accessory",
        "color": [
            188,
            208,
            182
        ],
        "isthing": 1,
        "id": 32,
        "name": "tie"
    },
    {
        "supercategory": "accessory",
        "color": [
            0,
            220,
            176
        ],
        "isthing": 1,
        "id": 33,
        "name": "suitcase"
    },
    {
        "supercategory": "sports",
        "color": [
            255,
            99,
            164
        ],
        "isthing": 1,
        "id": 34,
        "name": "frisbee"
    },
    {
        "supercategory": "sports",
        "color": [
            92,
            0,
            73
        ],
        "isthing": 1,
        "id": 35,
        "name": "skis"
    },
    {
        "supercategory": "sports",
        "color": [
            133,
            129,
            255
        ],
        "isthing": 1,
        "id": 36,
        "name": "snowboard"
    },
    {
        "supercategory": "sports",
        "color": [
            78,
            180,
            255
        ],
        "isthing": 1,
        "id": 37,
        "name": "sports ball"
    },
    {
        "supercategory": "sports",
        "color": [
            0,
            228,
            0
        ],
        "isthing": 1,
        "id": 38,
        "name": "kite"
    },
    {
        "supercategory": "sports",
        "color": [
            174,
            255,
            243
        ],
        "isthing": 1,
        "id": 39,
        "name": "baseball bat"
    },
    {
        "supercategory": "sports",
        "color": [
            45,
            89,
            255
        ],
        "isthing": 1,
        "id": 40,
        "name": "baseball glove"
    },
    {
        "supercategory": "sports",
        "color": [
            134,
            134,
            103
        ],
        "isthing": 1,
        "id": 41,
        "name": "skateboard"
    },
    {
        "supercategory": "sports",
        "color": [
            145,
            148,
            174
        ],
        "isthing": 1,
        "id": 42,
        "name": "surfboard"
    },
    {
        "supercategory": "sports",
        "color": [
            255,
            208,
            186
        ],
        "isthing": 1,
        "id": 43,
        "name": "tennis racket"
    },
    {
        "supercategory": "kitchen",
        "color": [
            197,
            226,
            255
        ],
        "isthing": 1,
        "id": 44,
        "name": "bottle"
    },
    {
        "supercategory": "kitchen",
        "color": [
            171,
            134,
            1
        ],
        "isthing": 1,
        "id": 46,
        "name": "wine glass"
    },
    {
        "supercategory": "kitchen",
        "color": [
            109,
            63,
            54
        ],
        "isthing": 1,
        "id": 47,
        "name": "cup"
    },
    {
        "supercategory": "kitchen",
        "color": [
            207,
            138,
            255
        ],
        "isthing": 1,
        "id": 48,
        "name": "fork"
    },
    {
        "supercategory": "kitchen",
        "color": [
            151,
            0,
            95
        ],
        "isthing": 1,
        "id": 49,
        "name": "knife"
    },
    {
        "supercategory": "kitchen",
        "color": [
            9,
            80,
            61
        ],
        "isthing": 1,
        "id": 50,
        "name": "spoon"
    },
    {
        "supercategory": "kitchen",
        "color": [
            84,
            105,
            51
        ],
        "isthing": 1,
        "id": 51,
        "name": "bowl"
    },
    {
        "supercategory": "food",
        "color": [
            74,
            65,
            105
        ],
        "isthing": 1,
        "id": 52,
        "name": "banana"
    },
    {
        "supercategory": "food",
        "color": [
            166,
            196,
            102
        ],
        "isthing": 1,
        "id": 53,
        "name": "apple"
    },
    {
        "supercategory": "food",
        "color": [
            208,
            195,
            210
        ],
        "isthing": 1,
        "id": 54,
        "name": "sandwich"
    },
    {
        "supercategory": "food",
        "color": [
            255,
            109,
            65
        ],
        "isthing": 1,
        "id": 55,
        "name": "orange"
    },
    {
        "supercategory": "food",
        "color": [
            0,
            143,
            149
        ],
        "isthing": 1,
        "id": 56,
        "name": "broccoli"
    },
    {
        "supercategory": "food",
        "color": [
            179,
            0,
            194
        ],
        "isthing": 1,
        "id": 57,
        "name": "carrot"
    },
    {
        "supercategory": "food",
        "color": [
            209,
            99,
            106
        ],
        "isthing": 1,
        "id": 58,
        "name": "hot dog"
    },
    {
        "supercategory": "food",
        "color": [
            5,
            121,
            0
        ],
        "isthing": 1,
        "id": 59,
        "name": "pizza"
    },
    {
        "supercategory": "food",
        "color": [
            227,
            255,
            205
        ],
        "isthing": 1,
        "id": 60,
        "name": "donut"
    },
    {
        "supercategory": "food",
        "color": [
            147,
            186,
            208
        ],
        "isthing": 1,
        "id": 61,
        "name": "cake"
    },
    {
        "supercategory": "furniture",
        "color": [
            153,
            69,
            1
        ],
        "isthing": 1,
        "id": 62,
        "name": "chair"
    },
    {
        "supercategory": "furniture",
        "color": [
            3,
            95,
            161
        ],
        "isthing": 1,
        "id": 63,
        "name": "couch"
    },
    {
        "supercategory": "furniture",
        "color": [
            163,
            255,
            0
        ],
        "isthing": 1,
        "id": 64,
        "name": "potted plant"
    },
    {
        "supercategory": "furniture",
        "color": [
            119,
            0,
            170
        ],
        "isthing": 1,
        "id": 65,
        "name": "bed"
    },
    {
        "supercategory": "furniture",
        "color": [
            0,
            182,
            199
        ],
        "isthing": 1,
        "id": 67,
        "name": "dining table"
    },
    {
        "supercategory": "furniture",
        "color": [
            0,
            165,
            120
        ],
        "isthing": 1,
        "id": 70,
        "name": "toilet"
    },
    {
        "supercategory": "electronic",
        "color": [
            183,
            130,
            88
        ],
        "isthing": 1,
        "id": 72,
        "name": "tv"
    },
    {
        "supercategory": "electronic",
        "color": [
            95,
            32,
            0
        ],
        "isthing": 1,
        "id": 73,
        "name": "laptop"
    },
    {
        "supercategory": "electronic",
        "color": [
            130,
            114,
            135
        ],
        "isthing": 1,
        "id": 74,
        "name": "mouse"
    },
    {
        "supercategory": "electronic",
        "color": [
            110,
            129,
            133
        ],
        "isthing": 1,
        "id": 75,
        "name": "remote"
    },
    {
        "supercategory": "electronic",
        "color": [
            166,
            74,
            118
        ],
        "isthing": 1,
        "id": 76,
        "name": "keyboard"
    },
    {
        "supercategory": "electronic",
        "color": [
            219,
            142,
            185
        ],
        "isthing": 1,
        "id": 77,
        "name": "cell phone"
    },
    {
        "supercategory": "appliance",
        "color": [
            79,
            210,
            114
        ],
        "isthing": 1,
        "id": 78,
        "name": "microwave"
    },
    {
        "supercategory": "appliance",
        "color": [
            178,
            90,
            62
        ],
        "isthing": 1,
        "id": 79,
        "name": "oven"
    },
    {
        "supercategory": "appliance",
        "color": [
            65,
            70,
            15
        ],
        "isthing": 1,
        "id": 80,
        "name": "toaster"
    },
    {
        "supercategory": "appliance",
        "color": [
            127,
            167,
            115
        ],
        "isthing": 1,
        "id": 81,
        "name": "sink"
    },
    {
        "supercategory": "appliance",
        "color": [
            59,
            105,
            106
        ],
        "isthing": 1,
        "id": 82,
        "name": "refrigerator"
    },
    {
        "supercategory": "indoor",
        "color": [
            142,
            108,
            45
        ],
        "isthing": 1,
        "id": 84,
        "name": "book"
    },
    {
        "supercategory": "indoor",
        "color": [
            196,
            172,
            0
        ],
        "isthing": 1,
        "id": 85,
        "name": "clock"
    },
    {
        "supercategory": "indoor",
        "color": [
            95,
            54,
            80
        ],
        "isthing": 1,
        "id": 86,
        "name": "vase"
    },
    {
        "supercategory": "indoor",
        "color": [
            128,
            76,
            255
        ],
        "isthing": 1,
        "id": 87,
        "name": "scissors"
    },
    {
        "supercategory": "indoor",
        "color": [
            201,
            57,
            1
        ],
        "isthing": 1,
        "id": 88,
        "name": "teddy bear"
    },
    {
        "supercategory": "indoor",
        "color": [
            246,
            0,
            122
        ],
        "isthing": 1,
        "id": 89,
        "name": "hair drier"
    },
    {
        "supercategory": "indoor",
        "color": [
            191,
            162,
            208
        ],
        "isthing": 1,
        "id": 90,
        "name": "toothbrush"
    },
    {
        "supercategory": "textile",
        "color": [
            255,
            255,
            128
        ],
        "isthing": 0,
        "id": 92,
        "name": "banner"
    },
    {
        "supercategory": "textile",
        "color": [
            147,
            211,
            203
        ],
        "isthing": 0,
        "id": 93,
        "name": "blanket"
    },
    {
        "supercategory": "building",
        "color": [
            150,
            100,
            100
        ],
        "isthing": 0,
        "id": 95,
        "name": "bridge"
    },
    {
        "supercategory": "raw-material",
        "color": [
            168,
            171,
            172
        ],
        "isthing": 0,
        "id": 100,
        "name": "cardboard"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            146,
            112,
            198
        ],
        "isthing": 0,
        "id": 107,
        "name": "counter"
    },
    {
        "supercategory": "textile",
        "color": [
            210,
            170,
            100
        ],
        "isthing": 0,
        "id": 109,
        "name": "curtain"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            92,
            136,
            89
        ],
        "isthing": 0,
        "id": 112,
        "name": "door-stuff"
    },
    {
        "supercategory": "floor",
        "color": [
            218,
            88,
            184
        ],
        "isthing": 0,
        "id": 118,
        "name": "floor-wood"
    },
    {
        "supercategory": "plant",
        "color": [
            241,
            129,
            0
        ],
        "isthing": 0,
        "id": 119,
        "name": "flower"
    },
    {
        "supercategory": "food-stuff",
        "color": [
            217,
            17,
            255
        ],
        "isthing": 0,
        "id": 122,
        "name": "fruit"
    },
    {
        "supercategory": "ground",
        "color": [
            124,
            74,
            181
        ],
        "isthing": 0,
        "id": 125,
        "name": "gravel"
    },
    {
        "supercategory": "building",
        "color": [
            70,
            70,
            70
        ],
        "isthing": 0,
        "id": 128,
        "name": "house"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            255,
            228,
            255
        ],
        "isthing": 0,
        "id": 130,
        "name": "light"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            154,
            208,
            0
        ],
        "isthing": 0,
        "id": 133,
        "name": "mirror-stuff"
    },
    {
        "supercategory": "structural",
        "color": [
            193,
            0,
            92
        ],
        "isthing": 0,
        "id": 138,
        "name": "net"
    },
    {
        "supercategory": "textile",
        "color": [
            76,
            91,
            113
        ],
        "isthing": 0,
        "id": 141,
        "name": "pillow"
    },
    {
        "supercategory": "ground",
        "color": [
            255,
            180,
            195
        ],
        "isthing": 0,
        "id": 144,
        "name": "platform"
    },
    {
        "supercategory": "ground",
        "color": [
            106,
            154,
            176
        ],
        "isthing": 0,
        "id": 145,
        "name": "playingfield"
    },
    {
        "supercategory": "ground",
        "color": [
            230,
            150,
            140
        ],
        "isthing": 0,
        "id": 147,
        "name": "railroad"
    },
    {
        "supercategory": "water",
        "color": [
            60,
            143,
            255
        ],
        "isthing": 0,
        "id": 148,
        "name": "river"
    },
    {
        "supercategory": "ground",
        "color": [
            128,
            64,
            128
        ],
        "isthing": 0,
        "id": 149,
        "name": "road"
    },
    {
        "supercategory": "building",
        "color": [
            92,
            82,
            55
        ],
        "isthing": 0,
        "id": 151,
        "name": "roof"
    },
    {
        "supercategory": "ground",
        "color": [
            254,
            212,
            124
        ],
        "isthing": 0,
        "id": 154,
        "name": "sand"
    },
    {
        "supercategory": "water",
        "color": [
            73,
            77,
            174
        ],
        "isthing": 0,
        "id": 155,
        "name": "sea"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            255,
            160,
            98
        ],
        "isthing": 0,
        "id": 156,
        "name": "shelf"
    },
    {
        "supercategory": "ground",
        "color": [
            255,
            255,
            255
        ],
        "isthing": 0,
        "id": 159,
        "name": "snow"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            104,
            84,
            109
        ],
        "isthing": 0,
        "id": 161,
        "name": "stairs"
    },
    {
        "supercategory": "building",
        "color": [
            169,
            164,
            131
        ],
        "isthing": 0,
        "id": 166,
        "name": "tent"
    },
    {
        "supercategory": "textile",
        "color": [
            225,
            199,
            255
        ],
        "isthing": 0,
        "id": 168,
        "name": "towel"
    },
    {
        "supercategory": "wall",
        "color": [
            137,
            54,
            74
        ],
        "isthing": 0,
        "id": 171,
        "name": "wall-brick"
    },
    {
        "supercategory": "wall",
        "color": [
            135,
            158,
            223
        ],
        "isthing": 0,
        "id": 175,
        "name": "wall-stone"
    },
    {
        "supercategory": "wall",
        "color": [
            7,
            246,
            231
        ],
        "isthing": 0,
        "id": 176,
        "name": "wall-tile"
    },
    {
        "supercategory": "wall",
        "color": [
            107,
            255,
            200
        ],
        "isthing": 0,
        "id": 177,
        "name": "wall-wood"
    },
    {
        "supercategory": "water",
        "color": [
            58,
            41,
            149
        ],
        "isthing": 0,
        "id": 178,
        "name": "water-other"
    },
    {
        "supercategory": "window",
        "color": [
            183,
            121,
            142
        ],
        "isthing": 0,
        "id": 180,
        "name": "window-blind"
    },
    {
        "supercategory": "window",
        "color": [
            255,
            73,
            97
        ],
        "isthing": 0,
        "id": 181,
        "name": "window-other"
    },
    {
        "supercategory": "plant",
        "color": [
            107,
            142,
            35
        ],
        "isthing": 0,
        "id": 184,
        "name": "tree-merged"
    },
    {
        "supercategory": "structural",
        "color": [
            190,
            153,
            153
        ],
        "isthing": 0,
        "id": 185,
        "name": "fence-merged"
    },
    {
        "supercategory": "ceiling",
        "color": [
            146,
            139,
            141
        ],
        "isthing": 0,
        "id": 186,
        "name": "ceiling-merged"
    },
    {
        "supercategory": "sky",
        "color": [
            70,
            130,
            180
        ],
        "isthing": 0,
        "id": 187,
        "name": "sky-other-merged"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            134,
            199,
            156
        ],
        "isthing": 0,
        "id": 188,
        "name": "cabinet-merged"
    },
    {
        "supercategory": "furniture-stuff",
        "color": [
            209,
            226,
            140
        ],
        "isthing": 0,
        "id": 189,
        "name": "table-merged"
    },
    {
        "supercategory": "floor",
        "color": [
            96,
            36,
            108
        ],
        "isthing": 0,
        "id": 190,
        "name": "floor-other-merged"
    },
    {
        "supercategory": "ground",
        "color": [
            96,
            96,
            96
        ],
        "isthing": 0,
        "id": 191,
        "name": "pavement-merged"
    },
    {
        "supercategory": "solid",
        "color": [
            64,
            170,
            64
        ],
        "isthing": 0,
        "id": 192,
        "name": "mountain-merged"
    },
    {
        "supercategory": "plant",
        "color": [
            152,
            251,
            152
        ],
        "isthing": 0,
        "id": 193,
        "name": "grass-merged"
    },
    {
        "supercategory": "ground",
        "color": [
            208,
            229,
            228
        ],
        "isthing": 0,
        "id": 194,
        "name": "dirt-merged"
    },
    {
        "supercategory": "raw-material",
        "color": [
            206,
            186,
            171
        ],
        "isthing": 0,
        "id": 195,
        "name": "paper-merged"
    },
    {
        "supercategory": "food-stuff",
        "color": [
            152,
            161,
            64
        ],
        "isthing": 0,
        "id": 196,
        "name": "food-other-merged"
    },
    {
        "supercategory": "building",
        "color": [
            116,
            112,
            0
        ],
        "isthing": 0,
        "id": 197,
        "name": "building-other-merged"
    },
    {
        "supercategory": "solid",
        "color": [
            0,
            114,
            143
        ],
        "isthing": 0,
        "id": 198,
        "name": "rock-merged"
    },
    {
        "supercategory": "wall",
        "color": [
            102,
            102,
            156
        ],
        "isthing": 0,
        "id": 199,
        "name": "wall-other-merged"
    },
    {
        "supercategory": "textile",
        "color": [
            250,
            141,
            255
        ],
        "isthing": 0,
        "id": 200,
        "name": "rug-merged"
    }
]