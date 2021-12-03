#from ..builder import DETECTORS
#from .detr import DETR


from mmdet.models.detectors.detr import DETR
from mmdet.models.builder import DETECTORS
from easymd.models.detectors.detr_w_mask import DETR_w_mask
from easymd.models.detectors.detr_plus import DETR_plus

@DETECTORS.register_module()
class TIS(DETR_w_mask):

    def __init__(self, *args, **kwargs):
        super(DETR_w_mask, self).__init__(*args, **kwargs)
@DETECTORS.register_module()
class TIS_plus(DETR_plus):

    def __init__(self, *args, **kwargs):
        super(DETR_plus, self).__init__(*args, **kwargs)
        self.count=0
