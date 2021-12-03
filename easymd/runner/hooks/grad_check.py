from mmcv.runner.hooks.hook import HOOKS, Hook
import torch

@HOOKS.register_module()
class GradChecker(Hook):
    def __init__(self) -> None:
        super().__init__()
    def after_train_iter(self,runner):
        for key,val in runner.model.named_parameters():
            if val.grad == None and  val.requires_grad:
                print('WARNNING: {key}\'s parameters are not be used!!!!'.format(key=key))


@HOOKS.register_module()
class CacheCleaner(Hook):
    def __init__(self) -> None:
        super().__init__()
    def after_train_epoch(self,runner):
        torch.cuda.empty_cache()
