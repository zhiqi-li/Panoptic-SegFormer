
num_query = 400
num_class = 133
from mmdet.core import bbox
import numpy as np
import torch
import cv2 as cv
from torch.nn.functional import softmax
import torchvision


import json 
all_things = 0
all_stuff = 0
with open('./datasets/annotations/panoptic_val2017_detection_format.json','r') as f:
    data = json.load(f)
    print(len(data['annotations']))
    for each in data['annotations']:
        if each['category_id']<=80:
            all_things+=1
        else:
            all_stuff+=1
print(all_things,all_stuff,all_things/(all_things+all_stuff))
map = np.zeros([400,133])
things_stuff_list = []
for i in range(num_query):
    with open('./query/{i}.txt'.format(i=i)) as f:
        img  = torch.ones([500,500,3]).numpy()*255
        things = 0
        stuff = 0
        for line in f.readlines():

            data = line.strip().split(' ')
            t= int (data[0])
            if t<80:
                things+=1
            else:
                stuff+=1

            cx, cy, w, h, bbox_area, mask_area = float(data[1]), float(data[2]), float(data[3]), float(data[4]), data[5],int(data[6])
            bbox_area = float(bbox_area[7:-1])
            cx, cy, w, h = int(500*cx), int(500*cy), int(500*w+0.5), int(500*h+0.5)
            #cv.drawKeypoints()
            '''
            if w/h>1.5: # bbox_area<=322:
                cv.circle(img,  (cx,cy), 2, color=(255,0,0), thickness=1)
            elif w/h<0.7: #322<bbox_area<962:
                cv.circle(img,  (cx,cy), 2, color=(0,255,0), thickness=1)
            else:
                cv.circle(img,  (cx,cy), 2, color=(0,0,255), thickness=1)'''
            if t<80:
                if mask_area<=322:
                    cv.circle(img,  (cx,cy), 2, color=(255,0,0), thickness=1)
                elif 322<mask_area<962:
                    cv.circle(img,  (cx,cy), 2, color=(0,255,0), thickness=1)
                else:
                    cv.circle(img,  (cx,cy), 2, color=(0,0,255), thickness=1)
            else:
                cv.circle(img,  (cx,cy), 2, color=(100,0,100), thickness=1)
            
            color = (w/h*50+100,0,0)
            
            map[i][t]+=1
        things_stuff_list.append(things/(things+stuff))
        print(i,things/(things+stuff),things+stuff)
        torchvision.utils.save_image(torch.tensor(img).permute(2,0,1), '{i}.png'.format(i=i))
map = torch.tensor(map)
#map =map.permute(1,0)
import matplotlib.pyplot as plt
import matplotlib
print('mean',np.mean(np.array(things_stuff_list)))
plt.hist(np.array(things_stuff_list), bins=20)
#print(map[0])
plt.show()
#print(map.sum(-1))
#print(map.shape)
#print(map[:,0])
for i in range(133):
    max = map[:,i].max()
    min = map[:,i].min()
    #print(max,min)
    map[:, i] = (map[:,i] - min)/(max-min)



import mmcv
mmcv.imshow(map.numpy())