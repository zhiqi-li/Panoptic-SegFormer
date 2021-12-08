from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
#!/usr/bin/env python
'''
This script converts panoptic COCO format to detection COCO format. More
information about the formats can be found here:
http://cocodataset.org/#format-data. All segments will be stored in RLE format.
Additional option:
- using option '--things_only' the script can discard all stuff
segments, saving segments of things classes only.
'''

import os, sys
import argparse
import numpy as np
import json
import time
import multiprocessing

import PIL.Image as Image

import functools
import traceback
import json
import numpy as np
from skimage.measure import label

# The decorator is used to prints an error trhown inside process
def get_traceback(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print('Caught exception in worker thread:')
            traceback.print_exc()
            raise e

    return wrapper


class IdGenerator():
    '''
    The class is designed to generate unique IDs that have meaningful RGB encoding.
    Given semantic category unique ID will be generated and its RGB encoding will
    have color close to the predefined semantic category color.
    The RGB encoding used is ID = R * 256 * G + 256 * 256 + B.
    Class constructor takes dictionary {id: category_info}, where all semantic
    class ids are presented and category_info record is a dict with fields
    'isthing' and 'color'
    '''
    def __init__(self, categories):
        self.taken_colors = set([0, 0, 0])
        self.categories = categories
        for category in self.categories.values():
            if category['isthing'] == 0:
                self.taken_colors.add(tuple(category['color']))

    def get_color(self, cat_id):
        def random_color(base, max_dist=30):
            new_color = base + np.random.randint(low=-max_dist,
                                                 high=max_dist+1,
                                                 size=3)
            return tuple(np.maximum(0, np.minimum(255, new_color)))

        category = self.categories[cat_id]
        if category['isthing'] == 0:
            return category['color']
        base_color_array = category['color']
        base_color = tuple(base_color_array)
        if base_color not in self.taken_colors:
            self.taken_colors.add(base_color)
            return base_color
        else:
            while True:
                color = random_color(base_color_array)
                if color not in self.taken_colors:
                    self.taken_colors.add(color)
                    return color

    def get_id(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color)

    def get_id_and_color(self, cat_id):
        color = self.get_color(cat_id)
        return rgb2id(color), color


def rgb2id(color):
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.int32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return int(color[0] + 256 * color[1] + 256 * 256 * color[2])


def id2rgb(id_map):
    if isinstance(id_map, np.ndarray):
        id_map_copy = id_map.copy()
        rgb_shape = tuple(list(id_map.shape) + [3])
        rgb_map = np.zeros(rgb_shape, dtype=np.uint8)
        for i in range(3):
            rgb_map[..., i] = id_map_copy % 256
            id_map_copy //= 256
        return rgb_map
    color = []
    for _ in range(3):
        color.append(id_map % 256)
        id_map //= 256
    return color


def save_json(d, file):
    with open(file, 'w') as f:
        json.dump(d, f)

try:
    # set up path for pycocotools
    # sys.path.append('./cocoapi-master/PythonAPI/')
    from pycocotools import mask as COCOmask
except Exception:
    raise Exception("Please install pycocotools module from https://github.com/cocodataset/cocoapi")

def center_of_mass(bitmasks):
    _, h, w = bitmasks.shape

    
    radius = []
    corner = []
    for mask in bitmasks:
        mask = np.expand_dims(mask, axis=2)
        rle = COCOmask.encode(np.asfortranarray(mask))[0]
        bbox = COCOmask.toBbox(rle)
        x,y,w,h = bbox
        radius.append(np.sqrt(w*h))
        corner.append([[x,y],[x+w,y],[x,y+h],[x+w,y+h]])
    return np.array(corner),np.array(radius)
def dist_ab(a,b):
    return np.sqrt(((a[0]-b[0])**2+ (a[1]-b[1])**2))
    
def min_dist(corner_m,corner_n):
    min_d = 100000

    for a in corner_m:
        for b in corner_n:
            min_d=min(dist_ab(a,b),min_d)

    return min_d
@get_traceback
def convert_panoptic_to_detection_coco_format_single_core(
    proc_id, annotations_set, categories, segmentations_folder, things_only,sub_stuff
):
    annotations_detection = []
    for working_idx, annotation in enumerate(annotations_set):
        if working_idx % 100 == 0:
            print('Core: {}, {} from {} images processed'.format(proc_id,
                                                                 working_idx,
                                                                 len(annotations_set)))

        
        file_name = '{}.png'.format(annotation['file_name'].rsplit('.')[0])
        try:
            pan_format = np.array(
                Image.open(os.path.join(segmentations_folder, file_name)), dtype=np.uint32
            )
        except IOError:
            raise KeyError('no prediction png file for id: {}'.format(annotation['image_id']))
        id_and_category_maps = rgb2id(pan_format)

        for segm_info in annotation['segments_info']:
            if things_only and categories[segm_info['category_id']]['isthing'] != 1:
                continue
            mask = (id_and_category_maps == segm_info['id']).astype(np.uint8)
            segm_info.pop('id')
            _mask = np.expand_dims(mask, axis=2)
            
            new_segm_info = segm_info.copy()
            new_segm_info['image_id'] = annotation['image_id']
            rle = COCOmask.encode(np.asfortranarray(_mask))[0]
            rle['counts'] = rle['counts'].decode('utf8')
            new_segm_info['segmentation'] = rle
            annotations_detection.append(new_segm_info)
                
            if sub_stuff and categories[segm_info['category_id']]['isthing'] != 1:
                
                sub_masks,num = label(mask,background=0,return_num=True)
                sub_bitmasks = []
                for i in range(1, num+1): # 这里从1开始，防止将背景设置为最大连通域
                    sub_bitmasks.append((sub_masks == i).astype(np.uint8))
                
                sub_bitmasks = np.stack(sub_bitmasks)
                corner,radius = center_of_mass(sub_bitmasks)
                index = np.argsort(radius)
                radius = radius[index]
                corner = corner[index]
                sub_bitmasks = sub_bitmasks[index]
                dist_matrix = np.eye(num)*100000
                for i in range(num-1):
                    for j in range(i+1,num):
                        dist_matrix[i][j] = min_dist(corner[i],corner[j])

                #d1,d2 = center[None,:],center[:,None]
                #dist_matrix = np.sqrt(((d1[...,0]-d2[...,0])**2+ (d1[...,1]-d2[...,1])**2))
                
                if num>4:
                    keep_max = 4
                    while radius[-keep_max]<32:
                        keep_max-=1
                        if keep_max==1:
                            break
                    while radius[-keep_max]>32 and keep_max<num:
                        keep_max+=1
                        
                    for i in range(num-keep_max):
                        dist_matrix[i][i] = 100000
                        nearest = np.argmin(dist_matrix[i,-keep_max:])
                        sub_bitmasks[-keep_max+nearest] |=sub_bitmasks[i]
                    sub_bitmasks = sub_bitmasks[-keep_max:]
                    num = keep_max
                
                corner,radius = center_of_mass(sub_bitmasks)
                index = np.argsort(radius)
                radius = radius[index]
                corner = corner[index]
                sub_bitmasks = sub_bitmasks[index]
                dist_matrix = np.eye(num)*100000
                for i in range(num-1):
                    for j in range(i+1,num):
                        dist_matrix[i][j] = min_dist(corner_m=corner[i],corner_n=corner[j])
                #dist_matrix = np.sqrt(((d1[...,0]-d2[...,0])**2+ (d1[...,1]-d2[...,1])**2))

                flag = np.ones([num])==1
                for i in range(num-1):
                    #for j in range(i+1,num):
                        dist_matrix[i][i] = 100000
                        nearest = np.argmin(dist_matrix[i,i+1:])+i+1
                        if dist_matrix[i][nearest]<(radius[i]+radius[nearest]) or radius[i]<32:
                            #print(i,nearest,dist_matrix[i][nearest],radius[i],radius[nearest])
                            sub_bitmasks[nearest] |= sub_bitmasks[i]
                            flag[i]=False

                            
                sub_bitmasks= sub_bitmasks[flag]
                
                for mask in sub_bitmasks:
                    mask = np.expand_dims(mask, axis=2)
                    #segm_info.pop('id')
                    new_segm_info = segm_info.copy()
                    new_segm_info['image_id'] = annotation['image_id']
                    rle = COCOmask.encode(np.asfortranarray(mask))[0]
                    new_segm_info['bbox'] = COCOmask.toBbox(rle).tolist()
                    new_segm_info['area'] = COCOmask.area(rle).tolist()
                    rle['counts'] = rle['counts'].decode('utf8')
                    new_segm_info['segmentation'] = rle
                    annotations_detection.append(new_segm_info)
            

    print('Core: {}, all {} images processed'.format(proc_id, len(annotations_set)))
    return annotations_detection


def convert_panoptic_to_detection_coco_format(input_json_file,
                                              segmentations_folder,
                                              output_json_file,
                                              categories_json_file,
                                              things_only,sub_stuff):
    start_time = time.time()

    if segmentations_folder is None:
        segmentations_folder = input_json_file.rsplit('.', 1)[0]

    print("CONVERTING...")
    print("COCO panoptic format:")
    print("\tSegmentation folder: {}".format(segmentations_folder))
    print("\tJSON file: {}".format(input_json_file))
    print("TO")
    print("COCO detection format")
    print("\tJSON file: {}".format(output_json_file))
    if things_only:
        print("Saving only segments of things classes.")
    print('\n')

    print("Reading annotation information from {}".format(input_json_file))
    with open(input_json_file, 'r') as f:
        d_coco = json.load(f)
    annotations_panoptic = d_coco['annotations']

    with open(categories_json_file, 'r') as f:
        categories_list = json.load(f)
    categories = {category['id']: category for category in categories_list}

    cpu_num = multiprocessing.cpu_count()
    annotations_split = np.array_split(annotations_panoptic, cpu_num)
    print("Number of cores: {}, images per core: {}".format(cpu_num, len(annotations_split[0])))
    workers = multiprocessing.Pool(processes=cpu_num)
    processes = []
    for proc_id, annotations_set in enumerate(annotations_split):
        p = workers.apply_async(convert_panoptic_to_detection_coco_format_single_core,
                                (proc_id, annotations_set, categories, segmentations_folder, things_only,sub_stuff))
        processes.append(p)
    annotations_coco_detection = []
    for p in processes:
        annotations_coco_detection.extend(p.get())
    for idx, ann in enumerate(annotations_coco_detection):
        ann['id'] = idx

    d_coco['annotations'] = annotations_coco_detection
    categories_coco_detection = []
    for category in d_coco['categories']:
        if things_only and category['isthing'] != 1:
            continue
        
        category.pop('isthing')
        try:
            category.pop('color')
        except:
            print(category.keys())
        categories_coco_detection.append(category)
        
    d_coco['categories'] = categories_coco_detection
    save_json(d_coco, output_json_file)

    t_delta = time.time() - start_time
    print("Time elapsed: {:0.2f} seconds".format(t_delta))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="The script converts panoptic COCO format to detection \
         COCO format. See this file's head for more information."
    )
    parser.add_argument('--input_json_file', type=str,
                        help="JSON file with panoptic COCO format")
    parser.add_argument(
        '--segmentations_folder', type=str, default=None, help="Folder with \
         panoptic COCO format segmentations. Default: X if input_json_file is \
         X.json"
    )
    parser.add_argument('--output_json_file', type=str,
                        help="JSON file with detection COCO format")
    parser.add_argument('--categories_json_file', type=str,
                        help="JSON file with Panoptic COCO categories information",
                        default='./panoptic_coco_categories.json')
    parser.add_argument('--things_only', action='store_true',
                        help="discard stuff classes")
    parser.add_argument('--sub_stuff', action='store_true',
                        help="discard stuff classes")

    args = parser.parse_args()
    convert_panoptic_to_detection_coco_format(args.input_json_file,
                                              args.segmentations_folder,
                                              args.output_json_file,
                                              args.categories_json_file,
                                              args.things_only,
                                              args.sub_stuff)
# python ./convert.py --input_json_file ./datasets/annotations/panoptic_val2017_detection_format.json  --segmentations_folder ./datasets/annotations/panoptic_val2017 --output_json_file test.json --sub_stuff