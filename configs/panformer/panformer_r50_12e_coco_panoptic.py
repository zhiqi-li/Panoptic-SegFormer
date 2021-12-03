
_base_ = './base.py'
lr_config = dict(policy='step', step=[8])
runner = dict(type='EpochBasedRunner', max_epochs=12)
