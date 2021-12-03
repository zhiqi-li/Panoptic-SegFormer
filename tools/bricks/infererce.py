from mmcv.runner import checkpoint
from mmdet.apis.inference import init_detector,LoadImage, inference_detector
import easymd

config = 'config.py'
#checkpoints = './checkpoints/pseg_r101_r50_latest.pth'
checkpoints = "path/to/pth"
img             = '000000322864.jpg'
results = {
    'img': './datasets/coco/val2017/'+img
}
model = init_detector(config,checkpoint=checkpoints)

results = inference_detector(model,'./datasets/coco/val2017/'+img)

