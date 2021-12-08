_base_ = './base.py'
_dim_ = 256
_num_levels_=4
model = dict(
    type='PanSeg',
    # get swin-large
    #import os
    #import torch
    #os.system('wget -O checkpoints/swinl.pth https://github.com/SwinTransformer/storage/releases/download/v1.0.0/swin_large_patch4_window7_224_22k.pth')
    #model = torch.load('checkpoints/swinl.pth')
    #torch.save(model['model'], 'checkpoints/swinl.pth')
    #print('DONE, swin-large was saved as checkpoints/swinl.pth')
    pretrained='./checkpoints/swinl.pth',
    backbone=dict(
        type='SwinTransformer',
        embed_dim=192,
        depths=[2, 2, 18, 2],
        num_heads=[6, 12, 24, 48],
        window_size=7,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.,
        attn_drop_rate=0.,
        drop_path_rate=0.3,
        ape=False,
        patch_norm=True,
        out_indices=(1, 2, 3),
        use_checkpoint=False),
    neck=dict(
        type='ChannelMapper',
        in_channels=[384, 768, 1536],
        kernel_size=1,
        out_channels=_dim_,
        act_cfg=None,
        norm_cfg=dict(type='GN', num_groups=32),
        num_outs=_num_levels_),
    bbox_head=dict(
        quality_threshold_things=0.3,
        quality_threshold_stuff=0.3,
    )
)
