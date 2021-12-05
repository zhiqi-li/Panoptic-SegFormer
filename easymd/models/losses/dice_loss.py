import torch
import math

import mmcv
import torch
import torch.nn as nn

from mmdet.core import bbox_overlaps
#from ..builder import LOSSES
#from .utils import weighted_loss
from mmdet.models.losses.utils import weighted_loss
from mmdet.models.builder import LOSSES
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



#@mmcv.jit(derivate=True, coderize=True)
@weighted_loss
def dice_loss(input, target,mask=None,eps=0.001):
    N,H,W = input.shape
    
    input = input.contiguous().view(N, H*W)
    target = target.contiguous().view(N, H*W).float()
    if mask is not None:
      mask = mask.contiguous().view(N, H*W).float()
      input = input * mask
      target = target * mask
    a = torch.sum(input * target, 1)
    b = torch.sum(input * input, 1) + eps
    c = torch.sum(target * target, 1) + eps
    d = (2 * a) / (b + c)
    #print('1-d max',(1-d).max())
    return 1 - d

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
    beta=1.0
    assert beta > 0
    assert pred.size() == target.size() and target.numel() > 0
    loss = torch.abs(pred - target)
    return loss






@LOSSES.register_module()
class DiceLoss(nn.Module):

    def __init__(self, eps=1e-6, reduction='mean', loss_weight=1.0):
        super(DiceLoss, self).__init__()
        self.eps = eps
        self.reduction = reduction
        self.loss_weight = loss_weight
        self.count = 0
    def forward(self,
                pred,
                target,
                weight=None,
                mask=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):

        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        #if weight is not None and weight.dim() > 1:
            # TODO: remove this in the future
            # reduce the weight of shape (n,w,h) to (n,) to match the
            # giou_loss of shape (n,)
            #assert weight.shape == pred.shape
            #weight = weight.mean((-2,-1))
        loss = self.loss_weight * dice_loss(
            pred,
            target,
            weight,
            mask=mask,
            eps=self.eps,
            reduction=reduction,
            avg_factor=avg_factor,
            **kwargs)
        #print('DiceLoss',loss, avg_factor)
        return loss



@LOSSES.register_module()
class BCEFocalLoss(torch.nn.Module):
  """
  二分类的Focalloss alpha 固定
  """
  def __init__(self, gamma=2, alpha=0.25, reduction='sum',loss_weight=1.0):
    super().__init__()
    self.gamma = gamma
    self.alpha = alpha
    self.reduction = reduction
    self.loss_weight = loss_weight
  def forward(self, _input, target):
    pt = torch.sigmoid(_input)

    #print(pt.shape, target.shape)
    alpha = self.alpha
    loss = - alpha * (1 - pt) ** self.gamma * target * torch.log(pt) - \
        (1 - alpha) * pt ** self.gamma * (1 - target) * torch.log(1 - pt)
    #print('loss_shape',loss.shape)
    if self.reduction == 'elementwise_mean':
      loss = torch.mean(loss)
    elif self.reduction == 'sum':
      loss = torch.sum(loss)
    
    return loss*self.loss_weight/54