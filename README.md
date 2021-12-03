# Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers


<div align="center">
  <img src="https://github.com/zhiqi-li/Panoptic-SegFormer/raw/master/figs/arch.png" width="100%" height="100%"/>
</div><br/>

## Results

| Backbone | Method | Lr Schd | PQ | Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| R-50  | Panoptic-SegFormer | 1x| 48.0 |[config](configs/panformer/panformer_r50_12e_coco_panoptic.py) | [model](https://download.openmmlab.com/mim-example/knet/det/knet/knet_s3_r50_fpn_1x_coco-panoptic/knet_s3_r50_fpn_1x_coco-panoptic_20211017_151750-395fbcba.pth) |
| R-50  | Panoptic-SegFormer | 2x| 49.6 |[config](configs/panformer/panformer_r50_24e_coco_panoptic.py) | [model](https://download.openmmlab.com/mim-example/knet/det/knet/knet_s3_r50_fpn_ms-3x_coco-panoptic/knet_s3_r50_fpn_ms-3x_coco-panoptic_20211017_054613-4375b8be.pth) |
| R-101  | Panoptic-SegFormer | 2x| 50.6 |[config](configs/panformer/panformer_r101_12e_coco_panoptic.py) | [model](https://download.openmmlab.com/mim-example/knet/det/knet/knet_s3_r101_fpn_ms-3x_coco-panoptic/knet_s3_r101_fpn_ms-3x_coco-panoptic_20211017_054501-9c600b0c.pth)  |
| PVTv2-B5  | Panoptic-SegFormer | 2x| 55.6 |[config](configs/panformer/panformer_pvtv2b5_24e_coco_panoptic.py) | [model](https://download.openmmlab.com/mim-example/knet/det/knet/knet_s3_swin-l_fpn_ms-3x_16x2_coco-panoptic/knet_s3_swin-l_fpn_ms-3x_16x2_coco-panoptic_20211020_062341-62f3bbff.pth) |
| Swin-L (window size 7)  | Panoptic-SegFormer | 2x| 55.8 |[config](configs/panformer/panformer_swinl_24e_coco_panoptic.py) | [model](https://download.openmmlab.com/mim-example/knet/det/knet/knet_s3_swin-l_fpn_ms-3x_16x2_coco-panoptic/knet_s3_swin-l_fpn_ms-3x_16x2_coco-panoptic_20211020_062341-62f3bbff.pth) |




## Install

###  Prerequisites

- Linux
- Python 3.6+
- PyTorch 1.5+
- torchvision
- CUDA 9.2+ (If you build PyTorch from source, CUDA 9.0 is also compatible)
- GCC 5+
- [mmcv-full==1.3.4](https://github.com/open-mmlab/mmcv/tree/v1.3.4)
- [mmdet==2.12.0](https://github.com/open-mmlab/mmdetection/tree/v2.12.0) # higher version may not work
- timm==0.4.5
- einops==0.3.0
- Pillow==8.0.1
- opencv-python==4.5.2

note: PyTorch1.8 has a bug in its [adamw.py](https://github.com/pytorch/pytorch/blob/v1.8.0/torch/optim/adamw.py) and it is solved in PyTorch1.9([see](https://github.com/pytorch/pytorch/blob/master/torch/optim/adamw.py)), you can easily solve it by comparing the difference.


### install easy-mmdet

```
python setup.py install 
```


## datasets 

When I began this project, mmdet dose not support panoptic segmentation officially. I convert the dataset from panoptic segmentation format to instance segmentation format for convenience.

### 1. prepare data (COCO)

```
cd easy-mmdet
mkdir datasets
cd datasets
ln -s path_to_coco coco
mkdir annotations/
cd annotations
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
unzip panoptic_annotations_trainval2017.zip
```

Then the directory structure should be the following:

```
easy-mmdet
├── datasets
│   ├── annotations/
│   │   ├── panoptic_train2017/
│   │   ├── panoptic_train2017.json
│   │   ├── panoptic_val2017/
│   │   └── panoptic_val2017.json
│   └── coco/ 
│
├── config
├── checkpoints
├── easymd
...
```

### 2. convert panoptic format to detection format 

```
cd easy-mmdet
./tools/convert_panoptic_coco.sh coco
```

Then the directory structure should be the following:

```
easy-mmdet
├── datasets
│   ├── annotations/
│   │   ├── panoptic_train2017/
│   │   ├── panoptic_train2017_detection_format.json
│   │   ├── panoptic_train2017.json
│   │   ├── panoptic_val2017/
│   │   ├── panoptic_val2017_detection_format.json
│   │   └── panoptic_val2017.json
│   └── coco/ 
│
├── config
├── checkpoints
├── easymd
...
```


## Run (panoptic segmentation)

### Train 

single-machine with 8 gpus.

```
./tools/dist_train.sh ./configs/panformer/panformer_r50_24e_coco_panoptic.py 8
```


### test

```
./tools/dist_test.sh ./configs/panformer/panformer_r50_24e_coco_panoptic.py path/to/model.pth 8
```

## <a name="Citing"></a>Citing

If you use Panoptic SegFormer in your research, please use the following BibTeX entry.

```BibTeX
@article{li2021panoptic,
  title={Panoptic SegFormer},
  author={Li, Zhiqi and Wang, Wenhai and Xie, Enze and Yu, Zhiding and Anandkumar, Anima and Alvarez, Jose M and Lu, Tong and Luo, Ping},
  journal={arXiv},
  year={2021}
}
```


## Acknowledgement

Mainly based on [Defromable DETR](https://github.com/open-mmlab/mmdetection.git). 

Thanks very much for other open source work: [timm](https://github.com/rwightman/pytorch-image-models), [Panoptic FCN] (https://github.com/dvlab-research/PanopticFCN)[MaskFomer](https://github.com/facebookresearch/MaskFormer)

