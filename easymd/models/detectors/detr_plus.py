from mmdet.core import bbox2result
#from ..builder import DETECTORS
from easymd.models.detectors.single_stage_panoptic_detector import SingleStagePanopticDetector
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
import torch
import numpy as np
from easymd.models.utils.transform import mask2result
from easymd.models.utils.visual import save_tensor
from mmdet.core import bbox2result, bbox_mapping_back
import mmcv
from torchvision.transforms.transforms import ToTensor
@DETECTORS.register_module()
class DETR_plus(SingleStagePanopticDetector):
    r"""Implementation of `DETR: End-to-End Object Detection with
    Transformers <https://arxiv.org/pdf/2005.12872>`_"""

    def __init__(self,
                 backbone,
                 neck=None,
                 bbox_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        
        super(DETR_plus, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained, init_cfg)
        self.count=0
    def simple_test(self, img, img_metas=None, rescale=False):
        """Test function without test time augmentation.

        Args:
            imgs (list[torch.Tensor]): List of multiple images
            img_metas (list[dict]): List of image information.
            rescale (bool, optional): Whether to rescale the results.
                Defaults to False.

        Returns:
            list[list[np.ndarray]]: BBox results of each image and classes.
                The outer list corresponds to each image. The inner list
                corresponds to each class.
        """
        
        batch_size = len(img_metas)
        assert batch_size == 1, 'Currently only batch_size 1 for inference ' \
            f'mode is supported. Found batch_size {batch_size}.'
        x = self.extract_feat(img)
        outs = self.bbox_head(x, img_metas)

        results = self.bbox_head.get_bboxes(*outs, img_metas, rescale=rescale)
        assert isinstance(results,dict), 'The return results should be a dict'
        
       
        results_dict = {}
        for return_type in results.keys():
            if return_type == 'bbox':
                labels = results['labels']
                bbox_list = results['bbox']
                bbox_results = [
                    bbox2result(det_bboxes, det_labels, self.bbox_head.num_things_classes)
                    for det_bboxes, det_labels in zip(bbox_list,labels)
                ]
                results_dict['bbox'] = bbox_results
            elif return_type == 'segm':
                seg_list = results['segm']
                labels = results['labels']
               
                masks_results = [
                    mask2result(det_segm,det_labels,self.bbox_head.num_things_classes)
                        for det_segm, det_labels in zip(seg_list,labels)
                ]
                results_dict['segm'] = masks_results
            elif return_type == 'panoptic':
                results_dict['panoptic'] = results['panoptic']

  

        
        
     
        return results_dict
  