import argparse
import os
from pathlib import Path

import mmdet
import mmcv
from mmcv import Config
from mmdet.datasets.builder import build_dataset
import random
import cv2 as cv
import numpy as np
import easymd
def parse_args():
    parser = argparse.ArgumentParser(description='Browse a dataset')
    parser.add_argument('config', help='train config file path')
    # 以下三个pipeline排除,方便可视化
    parser.add_argument(
        '--skip-type',
        type=str,
        nargs='+',
        default=['DefaultFormatBundle', 'Normalize', 'Collect'],
        help='skip some useless pipeline')
    parser.add_argument(
        '--output-dir',
        default=None,
        type=str,
        help='If there is no display interface, you can save it')
    parser.add_argument('--not-show', default=False, action='store_true')
    parser.add_argument(
        '--show-interval',
        type=int,
        default=0,
        help='the interval of show (ms)')
    args = parser.parse_args()
    return args


def retrieve_data_cfg(config_path, skip_type):
    cfg = Config.fromfile(config_path)
    train_data_cfg = cfg.data.train
    if train_data_cfg.get('dataset', None) is not None:
        # voc数据集
        datasets = train_data_cfg['dataset']
        datasets['pipeline'] = [
            x for x in datasets.pipeline if x['type'] not in skip_type
        ]
    else:
        train_data_cfg['pipeline'] = [
            x for x in train_data_cfg.pipeline if x['type'] not in skip_type
        ]

    return cfg


def main():
    args = parse_args()
    cfg = retrieve_data_cfg(args.config, args.skip_type)

    dataset = build_dataset(cfg.data.train)

    progress_bar = mmcv.ProgressBar(len(dataset))
    for item in dataset:
        #print(item.keys())
        filename = os.path.join(args.output_dir,
                                Path(item['filename']).name
                                ) if args.output_dir is not None else None
        img = mmcv.imshow_det_bboxes(
            item['img'],
            item['gt_bboxes'],
            item['gt_labels'],
            class_names=dataset.CLASSES,
            show=False,
            out_file=filename,
            wait_time=args.show_interval)
        img = img.astype('uint8')
        #print(dir(item['gt_masks'][0]))
        for each in item['gt_masks']:
            color = [random.randint(0,255),random.randint(0,255),random.randint(0,255)]
            each_3 = each[...,None] *color
            each_3 = each_3.astype('uint8')
            img[each==1] = (img[each==1]*0.4 + each_3[each==1]*0.6).astype('uint8')
            #np.clip(new_img_with_alpha,0,255)
        cv.imshow('img',img)
        if cv.waitKey(0) & 0xFF== ord('q'):
            exit(0)
        progress_bar.update()


if __name__ == '__main__':
    main()