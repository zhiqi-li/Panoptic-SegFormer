# Panoptic SegFormer: Delving Deeper into Panoptic Segmentation with Transformers
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/panoptic-segformer/panoptic-segmentation-on-coco-minival)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-minival?p=panoptic-segformer)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/panoptic-segformer/panoptic-segmentation-on-coco-test-dev)](https://paperswithcode.com/sota/panoptic-segmentation-on-coco-test-dev?p=panoptic-segformer)
<div align="center">
  <img src="https://github.com/zhiqi-li/Panoptic-SegFormer/raw/master/figs/arch.png" width="100%" height="100%"/>
</div><br/>

[arXiv](https://arxiv.org/abs/2109.03814)


## Results

results on COCO val

| Backbone | Method | Lr Schd | PQ | Config | Download |
| :---: | :---: | :---: | :---: | :---: | :---: |
| R-50  | Panoptic-SegFormer | 1x| 48.0 |[config](configs/panformer/panformer_r50_12e_coco_panoptic.py) | [model](https://github.com/zhiqi-li/Panoptic-SegFormer/releases/download/v1.0/panoptic_segformer_r50_1x.pth) |
| R-50  | Panoptic-SegFormer | 2x| 49.6 |[config](configs/panformer/panformer_r50_24e_coco_panoptic.py) | [model](https://github.com/zhiqi-li/Panoptic-SegFormer/releases/download/v1.0/panoptic_segformer_r50_2x.pth) |
| R-101  | Panoptic-SegFormer | 2x| 50.6 |[config](configs/panformer/panformer_r101_24e_coco_panoptic.py) | [model](https://github.com/zhiqi-li/Panoptic-SegFormer/releases/download/v1.0/panoptic_segformer_r101_2x.pth)  |
| [PVTv2-B5](https://github.com/whai362/PVT) (**much lighter**)  | Panoptic-SegFormer | 2x| 55.6 |[config](configs/panformer/panformer_pvtb5_24e_coco_panoptic.py) | [model](https://github.com/zhiqi-li/Panoptic-SegFormer/releases/download/v1.0/panoptic_segformer_pvtv2b5_2x.pth) |
| Swin-L (window size 7)  | Panoptic-SegFormer | 2x| 55.8 |[config](configs/panformer/panformer_swinl_24e_coco_panoptic.py) | [model](https://github.com/zhiqi-li/Panoptic-SegFormer/releases/download/v1.0/panoptic_segformer_swinl_2x.pth) |




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


### install Panoptic SegFormer

```
python setup.py install 
```


## Datasets 

When I began this project, mmdet dose not support panoptic segmentation officially. I convert the dataset from panoptic segmentation format to instance segmentation format for convenience.

### 1. prepare data (COCO)

```
cd Panoptic-SegFormer
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
Panoptic-SegFormer
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
cd Panoptic-SegFormer
./tools/convert_panoptic_coco.sh coco
```

Then the directory structure should be the following:

```
Panoptic-SegFormer
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

### train 

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

Mainly based on [Defromable DETR](https://github.com/open-mmlab/mmdetection.git) from MMdet. 

Thanks very much for other open source works: [timm](https://github.com/rwightman/pytorch-image-models), [Panoptic FCN](https://github.com/dvlab-research/PanopticFCN), [MaskFomer](https://github.com/facebookresearch/MaskFormer), [QueryInst](https://github.com/hustvl/QueryInst)


