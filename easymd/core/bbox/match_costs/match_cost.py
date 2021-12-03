from mmdet.models.losses.utils import weighted_loss
import torch

from mmdet.core.bbox.iou_calculators import bbox_overlaps
from mmdet.core.bbox.transforms import bbox_cxcywh_to_xyxy, bbox_xyxy_to_cxcywh
#from .builder import MATCH_COST
from mmdet.core.bbox.match_costs.builder import MATCH_COST
import torch.nn.functional as F
import mmcv
#from torchvision.utils import make_grid
from easymd.models.utils.visual import save_tensor
def center_of_mass(bitmasks):
    n, h, w = bitmasks.size()

    ys = torch.linspace(0, 1, h, dtype=torch.float32, device=bitmasks.device)
    xs = torch.linspace(0, 1, w, dtype=torch.float32, device=bitmasks.device)

    m00 = bitmasks.sum(dim=-1).sum(dim=-1).clamp(min=1e-6)
    m10 = (bitmasks * xs).sum(dim=-1).sum(dim=-1)
    m01 = (bitmasks * ys[:, None]).sum(dim=-1).sum(dim=-1)
    center_x = m10 / m00
    center_y = m01 / m00
    return torch.stack([center_x, center_y],-1)
    #return center_x, center_y

@weighted_loss
def l1_loss(pred, target):
    """Smooth L1 loss.

    Args:
        pred (torch.Tensor): The prediction.
        target (torch.Tensor): The learning target of the prediction.
        beta (float, optional): The threshold in the piecewise function.
            Defaults to 1.0.

    Returns:
        torch.Tensor: Calculated loss
    """

    #assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)

    return loss



@MATCH_COST.register_module()
class DiceCost(object):
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self,  weight=1.):
        self.weight = weight
        self.count =0 
    def __call__(self, input, target):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        #print('INPUT', input.shape)
        #print('target',target.shape)
        
        N1,H1,W1 = input.shape
        N2,H2,W2 = target.shape
     
        if H1!=H2 or W1!=W2:
            target = F.interpolate(target.unsqueeze(0),size=(H1,W1),mode='bilinear').squeeze(0)

        input = input.contiguous().view(N1, -1)[:,None,:]
        target = target.contiguous().view(N2, -1)[None,:,:]

        a = torch.sum(input * target, -1)
        b = torch.sum(input * input, -1) + 0.001
        c = torch.sum(target * target, -1) + 0.001
        d = (2 * a) / (b + c)
        return (1-d)*self.weight


@MATCH_COST.register_module()
class CenterCost(object):
    """IoUCost.

     Args:
         iou_mode (str, optional): iou mode such as 'iou' | 'giou'
         weight (int | float, optional): loss weight

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import IoUCost
         >>> import torch
         >>> self = IoUCost()
         >>> bboxes = torch.FloatTensor([[1,1, 2, 2], [2, 2, 3, 4]])
         >>> gt_bboxes = torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> self(bboxes, gt_bboxes)
         tensor([[-0.1250,  0.1667],
                [ 0.1667, -0.5000]])
    """

    def __init__(self,  weight=1.):
        self.weight = weight
        self.count =0 
    def __call__(self, input, target):
        """
        Args:
            bboxes (Tensor): Predicted boxes with unnormalized coordinates
                (x1, y1, x2, y2). Shape [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with unnormalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: iou_cost value with weight
        """
        # overlaps: [num_bboxes, num_gt]
        #print('INPUT', input.shape)
        #print('target',target.shape)
        
        N1,H1,W1 = input.shape
        N2,H2,W2 = target.shape
        if H1!=H2 or W1!=W2:
            target = F.interpolate(target.unsqueeze(0),size=(H1,W1),mode='bilinear').squeeze(0)
        #save_tensor(input,'{i}.png'.format(i=self.count))
        #self.count +=1
        input = center_of_mass(input)
        target = center_of_mass(target)
        input = input.contiguous().view(N1, 2)[:,None,:]
        target = target.contiguous().view(N2,2)[None,:,:]
        cost = l1_loss(input,target)
        
        return cost*self.weight



@MATCH_COST.register_module()
class BBoxL1Cost_center(object):
    """BBoxL1Cost.

     Args:
         weight (int | float, optional): loss_weight
         box_format (str, optional): 'xyxy' for DETR, 'xywh' for Sparse_RCNN

     Examples:
         >>> from mmdet.core.bbox.match_costs.match_cost import BBoxL1Cost
         >>> import torch
         >>> self = BBoxL1Cost()
         >>> bbox_pred = torch.rand(1, 4)
         >>> gt_bboxes= torch.FloatTensor([[0, 0, 2, 4], [1, 2, 3, 4]])
         >>> factor = torch.tensor([10, 8, 10, 8])
         >>> self(bbox_pred, gt_bboxes, factor)
         tensor([[1.6172, 1.6422]])
    """

    def __init__(self, weight=1., box_format='xyxy'):
        self.weight = weight
        assert box_format in ['xyxy', 'xywh']
        self.box_format = box_format

    def __call__(self, bbox_pred, gt_bboxes):
        """
        Args:
            bbox_pred (Tensor): Predicted boxes with normalized coordinates
                (cx, cy, w, h), which are all in range [0, 1]. Shape
                [num_query, 4].
            gt_bboxes (Tensor): Ground truth boxes with normalized
                coordinates (x1, y1, x2, y2). Shape [num_gt, 4].

        Returns:
            torch.Tensor: bbox_cost value with weight
        """
        if self.box_format == 'xywh':
            gt_bboxes = bbox_xyxy_to_cxcywh(gt_bboxes)
        elif self.box_format == 'xyxy':
            bbox_pred = bbox_cxcywh_to_xyxy(bbox_pred)
        bbox_cost = torch.cdist(bbox_pred[:,:2], gt_bboxes[:,:2], p=1)
        return bbox_cost * self.weight