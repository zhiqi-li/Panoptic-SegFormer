# dataset settings
dataset_type = 'mapillary_panoptic'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True),
    #dict(
    #        type='Resize',
    #        img_scale=[(512, 4096), (614, 4096), (716, 4096),
    #                       (819, 4096), (921, 4096), (1024, 4096),
    #                        (1126, 4096), (1228, 4096), (1221, 4096),
    #                       (1433, 4096), (1536, 4096), (1638, 4096), (1740, 4096),
    #                        (1843, 4096),(1945, 4096),(2048, 4096)],
    #            multiscale_mode='value',
    #            keep_ratio=True),

    dict(type='Resize', img_scale=(1024, 2048), ratio_range=(0.8, 2.0), keep_ratio=True),
    dict(type='RandomCrop', crop_size=(1024, 2048)),
    #dict(type='Resize', img_scale=(100, 100), ratio_range=(0.8, 2.0), keep_ratio=True),
    #dict(type='RandomCrop', crop_size=(100, 100)),
    #dict(
    #    type='Resize', img_scale=[(2048, 800), (2048, 1024)], keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=1),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks' ]),
]

test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1024, 2048),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=1),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=1,
    workers_per_gpu=3,
    train=dict(
        type=dataset_type,
        ann_file='./datasets/mapillary-vistas/train/panoptic_train_coco_format.json',
        img_prefix='./datasets/mapillary-vistas/train/images',
        pipeline=train_pipeline),
    #train=dict(
    #    type=dataset_type,
    #    ann_file= './datasets/ADEChallengeData2016/ade20k_panoptic_val_detection_format.json',
    #    img_prefix='./datasets/ADEChallengeData2016/images/validation',
    #    pipeline=train_pipeline),
    val=dict(
        img_info_file_folder='./datasets/mapillary-vistas/val/panoptic_val_coco_format.json',
        output_folder='seg',
        pred_json='pred.json',
        segmentations_folder='./seg',
        gt_json='./datasets/mapillary-vistas/val/panoptic_2018.json',
        gt_folder='./datasets/mapillary-vistas/val/panoptic',
        type=dataset_type,
        ann_file='./datasets/mapillary-vistas/val/panoptic_val_coco_format.json',
        img_prefix='./datasets/mapillary-vistas/val/images',
        pipeline=test_pipeline),
    test=dict(
        img_info_file_folder='./datasets/mapillary-vistas/val/panoptic_val_coco_format.json',
        output_folder='seg',
        pred_json='pred.json',
        segmentations_folder='./seg',
        gt_json='./datasets/mapillary-vistas/val/panoptic_2018.json',
        gt_folder='./datasets/mapillary-vistas/val/panoptic',
        type=dataset_type,
        ann_file='./datasets/mapillary-vistas/val/panoptic_val_coco_format.json',
        img_prefix='./datasets/mapillary-vistas/val/images',
        pipeline=test_pipeline),

        )
evaluation = dict(metric=[
    'panoptic'])
#evaluation = dict(interval=1, metric='bbox')