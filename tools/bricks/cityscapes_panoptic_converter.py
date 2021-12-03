#!/usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os, sys
import json
import glob
import numpy as np
import PIL.Image as Image
import argparse
from easymd.datasets.panopticapi.utils import IdGenerator, save_json
try:
    # set up path for cityscapes scripts
    # sys.path.append('./cityscapesScripts/')
    from cityscapesscripts.helpers.labels import labels, id2label
except Exception:
    raise Exception("Please load Cityscapes scripts from https://github.com/mcordts/cityscapesScripts")

#original_format_folder = '/home/data4/cityscapes/gtFine/test/'
# folder to store panoptic PNGs
#out_folder = './cityscapes_data/cityscapes_panoptic_test/'
# json with segmentations information
#out_file = './cityscapes_data/cityscapes_panoptic_test.json'

def panoptic_converter(original_format_folder, out_folder, out_file):

    if not os.path.isdir(out_folder):
        print("Creating folder {} for panoptic segmentation PNGs".format(out_folder))
        os.mkdir(out_folder)

    categories = []
    for idx, el in enumerate(labels):
        if el.ignoreInEval:
            continue
        
        categories.append({'id': el.id,
                           'name': el.name,
                           'color': el.color,
                           'supercategory': el.category,
                           'isthing': 1 if el.hasInstances else 0})

    categories_dict = {cat['id']: cat for cat in categories}

    file_list = sorted(glob.glob(os.path.join(original_format_folder, '*/*_gtFine_instanceIds.png')))

    images = []
    annotations = []
    for working_idx, f in enumerate(file_list):
        if working_idx % 10 == 0:
            print(working_idx, len(file_list))

        original_format = np.array(Image.open(f))

        file_name = f.split('/')[-1]
        image_id = file_name.rsplit('_', 2)[0]
        image_filename= '{}_leftImg8bit.png'.format(image_id)
        # image entry, id for image is its filename without extension
        images.append({"id": image_id,
                       "width": original_format.shape[1],
                       "height": original_format.shape[0],
                       "file_name": image_filename})
       
        pan_format = np.zeros((original_format.shape[0], original_format.shape[1], 3), dtype=np.uint8)
        id_generator = IdGenerator(categories_dict)

        idx = 0
        l = np.unique(original_format)
        segm_info = []
        for el in l:
            if el < 1000:
                semantic_id = el
                is_crowd = 1
            else:
                semantic_id = el // 1000
                is_crowd = 0
            if semantic_id not in categories_dict:
                continue
            if categories_dict[semantic_id]['isthing'] == 0:
                is_crowd = 0
            mask = original_format == el
            segment_id, color = id_generator.get_id_and_color(semantic_id)
            pan_format[mask] = color

            area = np.sum(mask) # segment area computation

            # bbox computation for a segment
            hor = np.sum(mask, axis=0)
            hor_idx = np.nonzero(hor)[0]
            x = hor_idx[0]
            width = hor_idx[-1] - x + 1
            vert = np.sum(mask, axis=1)
            vert_idx = np.nonzero(vert)[0]
            y = vert_idx[0]
            height = vert_idx[-1] - y + 1
            bbox = [int(x), int(y), int(width), int(height)]
            
            segm_info.append({"id": int(segment_id),
                              "category_id": int(semantic_id),
                              "area": int(area),
                              "bbox": bbox,
                              "iscrowd": is_crowd})

        annotations.append({'image_id': image_id,
                            'file_name': file_name,
                            "segments_info": segm_info})

        Image.fromarray(pan_format).save(os.path.join(out_folder, file_name))

    d = {'images': images,
         'annotations': annotations,
         'categories': categories,
        }

    save_json(d, out_file)
import os.path as osp
import mmcv
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script converts panoptic COCO format to detection \
         COCO format. See this file's head for more information."
    )
    parser.add_argument('--original_format_folder', type=str,
                        help="path/to/cityscapes/gtFine/")
   
    args = parser.parse_args()
   
    ## train 
    output_folder_base = './datasets/cityscapes'
    mmcv.mkdir_or_exist(output_folder_base)
    original_format_folder_train = osp.join(args.original_format_folder,'train')
    out_folder = osp.join(output_folder_base,'cityscapes_panoptic_train')
    panoptic_converter(original_format_folder_train, out_folder,out_folder+'.json')


    ## val
    original_format_folder_val = osp.join(args.original_format_folder,'val')
    out_folder = osp.join(output_folder_base,'cityscapes_panoptic_val')
    panoptic_converter(original_format_folder_val, out_folder,out_folder+'.json')

