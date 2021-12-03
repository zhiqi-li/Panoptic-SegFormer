from mmcv.runner import checkpoint
from mmdet.apis.inference import init_detector,LoadImage, inference_detector
import easymd
#config= './configs/psg/pseg_r101_50e.py'
#config = '/home/data2/lizhiqi/easy-mmdet/configs/ptt/ptt10_r50_16x2_12e_coco_panoptic.py'
#checkpoints = './checkpoints/pseg_r101_r50_latest.pth'
#checkpoints = "/home/data2/lizhiqi/easy-mmdet/checkpoints/ptt10_epoch_12.pth"

config = '/home/lzq/easy-mmdet/configs/ptt/ptt9_r101_16x2_24e_coco_panoptic.py'
#checkpoints = './checkpoints/pseg_r101_r50_latest.pth'
checkpoints = "/home/lzq/easy-mmdet/checkpoints/ptt9_r101_epoch_50.pth"
two_horse       = '000000364166.jpg'
car             = '000000322864.jpg'
sheep           = '000000397639.jpg'
two_horse2      = '000000124975.jpg'
two_chagnjinglu = '000000079188.jpg'
snow            = '000000000785.jpg'
football        = '000000022935.jpg'
many_cars       = '000000573943.jpg'
bear = '000000000285.jpg'
img = two_horse2
results = {
    'img': './datasets/coco/val2017/'+img
}
model = init_detector(config,checkpoint=checkpoints)

results = inference_detector(model,'./datasets/coco/val2017/'+img)

