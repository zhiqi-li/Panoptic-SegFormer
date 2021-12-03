
_base_ = './base.py'

lr_config = dict(policy='step', step=[18])
runner = dict(type='EpochBasedRunner', max_epochs=24)
