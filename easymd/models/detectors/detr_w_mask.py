from mmdet.core import bbox2result
#from ..builder import DETECTORS
from easymd.models.detectors.single_stage_w_mask import SingleStageDetector_w_mask
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
import torch
import numpy as np
from easymd.models.utils.transform import mask2result
from easymd.models.utils.visual import save_tensor


@DETECTORS.register_module()
class DETR_w_mask(SingleStageDetector_w_mask):
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
        
        super(DETR_w_mask, self).__init__(backbone, neck, bbox_head, train_cfg,
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
        bbox_list,seg_list = self.bbox_head.get_bboxes(
            *outs, img_metas, rescale=rescale)
        #print('bbox_list', len(bbox_list),len(bbox_list[0]),len(bbox_list[0][0]),len(bbox_list[0][1]))
        bbox_results = [
            bbox2result(det_bboxes, det_labels, self.bbox_head.num_classes)
            for det_bboxes, det_labels in bbox_list
        ]
        masks_results = [
            mask2result(seg,labels,self.bbox_head.num_classes)
            for seg,(bboxes,labels) in zip(seg_list,bbox_list)
        ]
        #return bbox_results
        result = list(zip(bbox_results, masks_results))
        '''
        for each in result:
            bbox, mask =each
            for i in range(80):
                if len(bbox[i])>0:
                    print(bbox[i].shape)
                    print(mask[i].shape)
                    save_tensor(torch.tensor(mask[i]),'mask_{i}_small.png'.format(i=i),bbox=torch.tensor(bbox[i][:,:-1]))
        exit(0)'''
        return result
