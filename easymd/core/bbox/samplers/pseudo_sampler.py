import torch

#from ..builder import BBOX_SAMPLERS
#from .base_sampler import BaseSampler
#from .sampling_result import SamplingResult
from mmdet.core.bbox.samplers.sampling_result import SamplingResult
from easymd.core.bbox.samplers.sampling_result import SamplingResult_w_mask,SamplingResult_w_mask2,SamplingResult_w_mask3
from mmdet.core.bbox.samplers.base_sampler import BaseSampler
from mmdet.core.bbox.builder import BBOX_SAMPLERS

@BBOX_SAMPLERS.register_module()
class PseudoSampler_w_mask(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes,masks,gt_masks, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult_w_mask(pos_inds, neg_inds, bboxes, gt_bboxes,masks,gt_masks,
                                         assign_result, gt_flags,**kwargs)
        return sampling_result
@BBOX_SAMPLERS.register_module()
class PseudoSampler_w_mask2(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, bboxes, gt_bboxes,gt_masks, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = bboxes.new_zeros(bboxes.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult_w_mask2(pos_inds, neg_inds, bboxes, gt_bboxes,gt_masks,
                                         assign_result, gt_flags,**kwargs)
        return sampling_result
@BBOX_SAMPLERS.register_module()
class PseudoSampler_w_mask3(BaseSampler):
    """A pseudo sampler that does not do sampling actually."""

    def __init__(self, **kwargs):
        pass

    def _sample_pos(self, **kwargs):
        """Sample positive samples."""
        raise NotImplementedError

    def _sample_neg(self, **kwargs):
        """Sample negative samples."""
        raise NotImplementedError

    def sample(self, assign_result, masks, gt_masks, **kwargs):
        """Directly returns the positive and negative indices  of samples.

        Args:
            assign_result (:obj:`AssignResult`): Assigned results
            bboxes (torch.Tensor): Bounding boxes
            gt_bboxes (torch.Tensor): Ground truth boxes

        Returns:
            :obj:`SamplingResult`: sampler results
        """
        pos_inds = torch.nonzero(
            assign_result.gt_inds > 0, as_tuple=False).squeeze(-1).unique()
        
        neg_inds = torch.nonzero(
            assign_result.gt_inds == 0, as_tuple=False).squeeze(-1).unique()
        gt_flags = masks.new_zeros(masks.shape[0], dtype=torch.uint8)
        sampling_result = SamplingResult_w_mask3(pos_inds, neg_inds, gt_masks,
                                         assign_result, gt_flags,**kwargs)
        return sampling_result