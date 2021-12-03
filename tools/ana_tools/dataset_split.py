import json 
import random

file_path = './datasets/cityscapes/cityscapes_panoptic_train_detection_format.json'

with open(file_path,'r') as f:
    data = json.load(f)
    images = data['images']
    len_img = len(images)
    print(len_img)
    perm = [i for i in range(len_img)]
    random.shuffle(images)
    print(images[:len_img//10])
    data['images'] =  data['images'][:len_img//10]
    with open('partial_')