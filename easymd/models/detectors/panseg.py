#from ..builder import DETECTORS
#from .detr import DETR


from mmdet.models.detectors.detr import DETR
from mmdet.models.builder import DETECTORS
from easymd.models.detectors.detr_plus import DETR_plus

class PanSeg(DETR_plus):

    def __init__(self, *args, **kwargs):
        super(DETR_plus, self).__init__(*args, **kwargs)
        self.count=0
