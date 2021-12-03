from mmdet.core import bbox2result
#from ..builder import DETECTORS
from easymd.models.detectors.single_stage_w_mask import SingleStageDetector_w_mask
from mmdet.models.builder import DETECTORS, build_backbone, build_head, build_neck
import torch
import numpy as np
from easymd.models.utils.transform import mask2result
from easymd.models.utils.visual import save_tensor
from mmdet.core import bbox2result, bbox_mapping_back
import mmcv
from torchvision.transforms.transforms import ToTensor
@DETECTORS.register_module()
class DETR_plus(SingleStageDetector_w_mask):
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

  
        #print('bbox_list', len(bbox_list),len(bbox_list[0]),len(bbox_list[0][0]),len(bbox_list[0][1]))
        
        #return bbox_results
        
        
        '''
        for each in result:
            bbox, mask =each
            for i in range(80):
                if len(bbox[i])>0:
                    print(bbox[i].shape)
                    print(mask[i].shape)
                    save_tensor(torch.tensor(mask[i]),'mask_{i}_small.png'.format(i=i),bbox=torch.tensor(bbox[i][:,:-1]))
        exit(0)'''
        return results_dict
    '''
    def aug_test(self, img, img_metas, rescale=False):
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
        results = []
        for img_i,img_metas_i in zip(img,img_metas):
            
            x = self.extract_feat(img_i)
            outs = self.bbox_head(x, img_metas_i)
            results_i = self.bbox_head.get_aug_bboxes(*outs, img_metas_i, rescale=rescale)
            results.append(results_i)
        results = self.bbox_head.merge_results(results,img_metas[0])
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

  
        #print('bbox_list', len(bbox_list),len(bbox_list[0]),len(bbox_list[0][0]),len(bbox_list[0][1]))
        
        #return bbox_result
        return results_dict'''
        
    
    def merge_aug_results(self, aug_results, img_metas):
        """Merge augmented detection bboxes and score.

        Args:
            aug_results (list[list[Tensor]]): Det_bboxes and det_labels of each
                image.
            img_metas (list[list[dict]]): Meta information of each image, e.g.,
                image size, scaling factor, etc.

        Returns:
            tuple: (bboxes, labels)
        """

        recovered_bboxes, aug_labels = [], []
        aug_masks = []
        aug_labels_sutff , aug_masks_stuff,aug_bbox_stuff = [], [], []
        for results, img_info in zip(aug_results, img_metas):
            img_shape = img_info[0]['img_shape']  # using shape before padding
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            #bboxes, labels = bboxes_labels
            things_bboxes = results['thing']['bbox_things']
            things_labels = results['thing']['labels_things']
            things_masks = results['thing']['mask_pred_things']
            if flip:
                things_masks = torch.flip(things_masks,(-1,))
            aug_labels_sutff.append(results['stuff']['labels_stuff'])
            if flip:
                rec_mask = torch.flip(results['stuff']['mask_pred_stuff'],(-1,))
                aug_masks_stuff.append(rec_mask)
            else:
                aug_masks_stuff.append(results['stuff']['mask_pred_stuff'])
            
            bboxes, scores = things_bboxes[:, :4], things_bboxes[:, -1:]
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(torch.cat([bboxes, scores], dim=-1))
            stuff_bboxes=results['stuff']['bbox_stuff']
            bboxes_s, scores_s = stuff_bboxes[:, :4], stuff_bboxes[:, -1:]
            bboxes_s = bbox_mapping_back(bboxes_s, img_shape, scale_factor, flip)
            aug_bbox_stuff.append(torch.cat([bboxes_s, scores_s], dim=-1))
            aug_labels.append(things_labels)
            aug_masks.append(things_masks)

        bboxes_things = torch.cat(recovered_bboxes, dim=0)
        labels_things = torch.cat(aug_labels)
        masks_things = torch.cat(aug_masks,0)

        labels_stuff = torch.cat(aug_labels_sutff)
        masks_stuff = torch.cat(aug_masks_stuff,0)
        bboxes_stuff = torch.cat(aug_bbox_stuff)

        if bboxes_things.shape[0] > 0:
            out_bboxes, out_labels,out_masks = self.bbox_head._nms(
                bboxes_things, labels_things,masks_things, self.bbox_head.test_cfg)
        else:
            out_bboxes, out_labels,out_masks = bboxes, labels_things, masks_things
        if bboxes_stuff.shape[0] > 0:
            bboxes_stuff, labels_stuff,masks_stuff = self.bbox_head._nms(
                    bboxes_stuff, labels_stuff,masks_stuff, self.bbox_head.test_cfg)
        infos = ( out_bboxes, out_labels,out_masks),(bboxes_stuff, labels_stuff,masks_stuff)
        panoptic_results = self.bbox_head.merge_results(infos,img_metas[0])
        '''
        num = len(out_masks)
        raw_img = mmcv.imread(img_metas[0][0]['filename'])[:,:,::-1].copy()
        raw_img = ToTensor()(raw_img).squeeze(0).repeat(num,1,1,1).to(out_masks.device)
        #raw_img = torchvision.io.read_image(img_metas[img_id]['filename']).squeeze(0).repeat(100,1,1,1).to(mask_pred.device)*1.
        #raw_img = F.interpolate(img[img_id].unsqueeze(0),size=ori_shape[:2],mode='bilinear').squeeze(0).repeat(100,1,1,1)
        output_mask_pred = out_masks.detach().to(torch.float)
        for i in range(num):
            mask = output_mask_pred[i]==1
            mask_color = output_mask_pred[i,...,None]*torch.tensor([0,255,0],device=output_mask_pred.device)
            mask_color = mask_color.permute(2,0,1)
            raw_img[i][:,mask] = torch.clamp(raw_img[i][:,mask]*0.8 + mask_color[:,mask]*0.2,0,255)
        
        save_tensor(raw_img,'mask_{i}.png'.format(i=self.count),labels=out_labels,scores=out_bboxes[:,-1],bbox=out_bboxes[:,:-1])
        self.count+=1'''
        results = {
            'labels': [out_labels],
            'bbox': [out_bboxes],
            'segm': [out_masks],
            'panoptic':[panoptic_results]
        }
        return results

    def aug_test(self, imgs, img_metas, rescale=False):



        #assert img_metas[0][0]['flip'] + img_metas[1][0]['flip'], (
        #    'aug test must have flipped image pair')
        img_inds = list(range(len(imgs)))
        aug_results = []
        for ind, flip_ind in zip(img_inds[0::2], img_inds[1::2]):
            img_pair = torch.cat([imgs[ind], imgs[flip_ind]])
            x = self.extract_feat(img_pair)
            outs = self.bbox_head(x,[img_metas[ind][0], img_metas[flip_ind][0]])
            bbox_list = self.bbox_head.get_aug_bboxes(
                *outs, [img_metas[ind][0], img_metas[flip_ind][0]], rescale=False)
            aug_results.append(bbox_list[0])
            aug_results.append(bbox_list[1])
        results = self.merge_aug_results(aug_results, img_metas)
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
