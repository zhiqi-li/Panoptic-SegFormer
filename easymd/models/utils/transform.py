import torch
import numpy as np

def mask2result(seg, labels, num_classes):
    """Convert detection results to a list of numpy arrays.

    Args:
        bboxes (torch.Tensor | np.ndarray): shape (n, 5)
        labels (torch.Tensor | np.ndarray): shape (n, )
        num_classes (int): class number, including background class

    Returns:
        list(ndarray): bbox results of each class
    """
    
    if seg.shape[0] == 0:
        _,h,w = seg.shape
        return [np.zeros((0, h, w), dtype=np.float32) for i in range(num_classes)]
    else:
        if isinstance(seg, torch.Tensor):
            seg = seg.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
        return [seg[labels == i, :] for i in range(num_classes)]