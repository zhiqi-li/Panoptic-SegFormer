#!/usr/bin/env python2
'''
Visualization demo for panoptic COCO sample_data

The code shows an example of color generation for panoptic data (with
"generate_new_colors" set to True). For each segment distinct color is used in
a way that it close to the color of corresponding semantic class.
'''
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import numpy as np
import json

import PIL.Image as Image
import matplotlib.pyplot as plt
from skimage.segmentation import find_boundaries
import mmcv
from panopticapi.utils import IdGenerator, rgb2id
try:
    from detectron2.data import MetadataCatalog
except:
    print('no detecteon2')
#from detectron2.utils.visualizer import Visualizer
import torch
from easymd.models.utils.visual import Visualizer # we modified the Visualizer from detectron2
# whether from the PNG are used or new colors are generated
generate_new_colors = True

json_file = './datasets/annotations/panoptic_val2017.json'
segmentations_folder = './datasets/annotations/panoptic_val2017'
img_folder = './datasets/coco/val2017'
panoptic_coco_categories = './easymd/datasets/panoptic_coco_categories.json'

with open(json_file, 'r') as f:
    coco_d = json.load(f)

#ann = np.random.choice(coco_d['annotations'])
#id = 785 
#id = 79188
#id = 124975 #two horses
#id = 2592
#id = 26564

def f(id):
    find = False
    for each in coco_d['annotations']:
        if each['image_id'] == id:
            ann = each
            find=True
            break
    if not find:
        return

    with open(panoptic_coco_categories, 'r') as f:
        categories_list = json.load(f)
    categegories = {category['id']: category for category in categories_list}

    # find input img that correspond to the annotation
    img = None
    for image_info in coco_d['images']:
        if image_info['id'] == ann['image_id']:
            try:
                img = np.array(
                    Image.open(os.path.join(img_folder, image_info['file_name']))
                )
            except:
                print("Undable to find correspoding input image.")
            break

    segmentation = np.array(
        Image.open(os.path.join(segmentations_folder, ann['file_name'])),
        dtype=np.uint8
    )
    segmentation_id = rgb2id(segmentation)
    # find segments boundaries


    if generate_new_colors:
        segmentation[:, :, :] = 0
        color_generator = IdGenerator(categegories)

        i =0

        for segment_info in ann['segments_info']:
            #print(segment_info)
            #if segment_info['id']!=  4475732:
            #    continue

            color = color_generator.get_color(segment_info['category_id'])
            mask = segmentation_id == segment_info['id']
            segmentation[mask] =color
            #print(dir(segment_info))
            
            segment_info.setdefault('isthing',True)
            #print(segment_info['category_id'] > 90,segment_info['category_id'])
            if segment_info['category_id'] > 90:
                segment_info['isthing'] =False
            i+=1
        if i<10:
            return

    boundaries = find_boundaries(rgb2id(segmentation), mode='thick')
    #segmentation[boundaries] = [0,255,0]
    # depict boundaries
    import cv2 as cv

    print(img.shape,segmentation.shape)
    #res = cv.add(segmentation,img)

    #im = Image.open(data['img_metas'][0].data[0][0]['filename'])
    meta = MetadataCatalog.get("coco_2017_val_panoptic_separated")
    im = np.array(img)[:, :, ::-1]
    v = Visualizer(im, meta, scale=1.0)
    v._default_font_size = 10
    v = v.draw_panoptic_seg_predictions(torch.from_numpy(segmentation_id), ann['segments_info'], area_threshold=0)
    res = v.get_image()[:,:,::-1]
    mmcv.imwrite(v.get_image(),'tmp.png')




    if img is None:
        plt.figure()
        plt.imshow(segmentation)
        plt.axis('off')
    else:
        plt.figure(figsize=(9, 5))
        plt.subplot(231)
        plt.imshow(img)
        plt.axis('off')
        plt.subplot(232)
        plt.imshow(segmentation)
        plt.axis('off')
        plt.subplot(233)
        plt.imshow(res)
        plt.axis('off')
        plt.subplot(234)
        msg = np.array(
            Image.open(os.path.join('/home/lzq/easy-mmdet/seg_pwm', ann['file_name'])),dtype=np.uint8
    )
        plt.imshow(msg)
        plt.axis('off')
        plt.subplot(235)
        pwm = np.array(
        Image.open(os.path.join('/home/lzq/easy-mmdet/seg_max', ann['file_name'])),dtype=np.uint8
    )
        plt.imshow(pwm)
        plt.axis('off')
        plt.subplot(236)
        hp = np.array(
        Image.open(os.path.join('/home/lzq/easy-mmdet/seg_hp', ann['file_name'])),
        dtype=np.uint8
    )
        plt.imshow(hp)
        plt.axis('off')
        plt.tight_layout()
    plt.show()
#{"mode":"full","isActive":false}
id=165681
#f_id(id)
f(id)
#while True:
#    f(id)
#    id+=1

    #boundaries = find_boundaries(rgb2id(segmentation), mode='thick')
    #mmcv.imwrite(segmentation[:,:,::-1],'gt/'+str(id)+'.png')
    #segmentation[boundaries] = [0,255,0]
    # depict boundaries

#for i in range(581781+1):
#    f(i)
'''
import cv2 as cv

print(img.shape,segmentation.shape)
res = cv.add(segmentation,img)



if img is None:
    plt.figure()
    plt.imshow(segmentation)
    plt.axis('off')
else:
    plt.figure(figsize=(9, 5))
    plt.subplot(131)
    plt.imshow(img)
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(segmentation)
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(res)
    plt.axis('off')
    plt.tight_layout()
plt.show()
#{"mode":"full","isActive":false}
'''

