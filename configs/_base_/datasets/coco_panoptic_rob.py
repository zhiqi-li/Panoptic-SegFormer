# dataset settings
dataset_type = 'CocoDataset_plus2'
data_root = 'datasets/coco/'
coco_root = 'datasets/'
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True, with_mask=True,with_seg=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels', 'gt_masks','gt_semantic_seg']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        ann_file= './datasets/annotations/panoptic_train2017_detection_format.json',
        img_prefix=data_root + 'train2017/',
        pipeline=train_pipeline),
    val=dict( 
        img_info_file_folder = './datasets/annotations/panoptic_val2017_detection_format.json',
        #img_info_file_folder = './datasets/coco/annotations/image_info_test-dev2017.json',
        output_folder = 'seg',
        pred_json = 'pred.json',
        segmentations_folder='./seg',
        gt_json = './datasets/annotations/panoptic_val2017.json',
        gt_folder = './datasets/annotations/panoptic_val2017',
        type=dataset_type,
        ann_file=data_root + 'annotations/instances_val2017.json',
        img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline),
    test=dict(
        #img_info_file_folder = './datasets/coco/annotations/image_info_test-dev2017.json',
        img_info_file_folder = './datasets/annotations/mini_panoptic_val.json',
        output_folder = 'seg_r50',
        pred_json = 'pred.json',
        segmentations_folder='./seg_swin',
        gt_json = './datasets/annotations/mini_panoptic_val2.json',
        gt_folder = './datasets/annotations/panoptic_val2017',
        type=dataset_type,
        #ann_file= './datasets/coco/annotations/image_info_test-dev2017.json',
        ann_file='./datasets/annotations/mini_inst_val.json',
        #img_prefix=data_root + 'test2017/',d
        img_prefix = '/home/lzq/coco_c/val_impulse_noise_1',
        # img_prefix=data_root + 'val2017/',
        pipeline=test_pipeline)
        )
evaluation = dict(metric=['bbox', 'segm', 'panoptic'])
#evaluation = dict(interval=1, metric='bbox')