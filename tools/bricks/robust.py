import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from imagecorruptions import corrupt
from PIL import Image
import seaborn as sns
import random
import cv2
import os
import mmcv
#from IPython import embed


random.seed(8) # for reproducibility
np.random.seed(8)

corruptions = ['gaussian_noise', 'shot_noise', 'impulse_noise', 'defocus_blur',
                'glass_blur', 'motion_blur', 'zoom_blur', 'snow', 'frost', 'fog',
                'brightness', 'contrast', 'elastic_transform', 'pixelate', 'jpeg_compression',
                'speckle_noise', 'gaussian_blur', 'spatter', 'saturate']
# corruption_dict = \
#     {'brightness': 'Brightness',
#      'contrast': 'Contrast' ,
#       'defocus_blur': 'Defocus Blur',
#       'elastic_transform': 'Elastic Transform',
#       'fog': 'Fog',
#       'frost': 'Frost',
#       'gaussian_noise': 'Gaussian Noise',
#       'glass_blur': 'Glass Blur',
#       'impulse_noise': 'Impulse Noise',
#       'jpeg_compression': 'JPEG Compression',
#       'motion_blur': 'Motion Blur',
#       'pixelate': 'Pixelate',
#       'shot_noise': 'Shot Noise',
#       'snow': 'Snow',
#       'zoom_blur': 'Zoom Blur',
#       'speckle_noise': 'Speckle Noise',
#       'gaussian_blur': 'Gaussian Blur',
#       'spatter': 'Spatter',
#       'saturate': 'Saturate'}

def perturb(i, p, s):
    img = corrupt(i, corruption_name=p, severity=s)
    return img


def convert_img_path(ori_path, suffix):
    new_path = ori_path.replace('val', 'val'+'_'+suffix)
    assert new_path != ori_path
    return new_path

def main():
    img_dir = '/home/data2/lizhiqi/coco_c/val'
    severity = [1, 2, 3, 4, 5]
    prog_bar = mmcv.ProgressBar(5000)
    for img_path in mmcv.scandir(img_dir, suffix='jpg', recursive=True):
        img_path = os.path.join(img_dir, img_path)
        clean_img_path = convert_img_path(img_path, 'clean')
        img = mmcv.imread(img_path)
        mmcv.imwrite(img, clean_img_path, auto_mkdir=True)
        prog_bar.update()
        for p in corruptions:
            for s in severity:
                perturbed_img = perturb(img, p, s)
                img_suffix = p+"_"+str(s)
                perturbed_img_path = convert_img_path(img_path, img_suffix)
                mmcv.imwrite(perturbed_img, perturbed_img_path, auto_mkdir=True)


if __name__ == '__main__':
    main()