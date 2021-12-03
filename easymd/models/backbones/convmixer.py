import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.registry import register_model
#from .helpers import build_model_with_cfg
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from mmcv.runner import _load_checkpoint, load_state_dict
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import  torch.nn.functional as F
from easymd.models.utils.visual import save_tensor
import re
def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .96, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD, 'classifier': 'head',
        'first_conv': 'stem.0',
        **kwargs
    }


default_cfgs = {
    'convmixer_1536_20': _cfg(url='https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_1536_20_ks9_p7.pth.tar'),
    'convmixer_768_32': _cfg(url='https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_768_32_ks7_p7_relu.pth.tar'),
    'convmixer_1024_20_ks9_p14': _cfg(url='https://github.com/tmp-iclr/convmixer/releases/download/timm-v1.0/convmixer_1024_20_ks9_p14.pth.tar')
}


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x):
        return self.fn(x) + x


class ConvMixer(nn.Module):
    def __init__(self, dim=768, depth=32, kernel_size=9, patch_size=7, in_chans=3,  activation=nn.GELU, pretrained=None,out_indices=[0,1,2,3], **kwargs):
        super().__init__()
        #self.num_classes = num_classes
        self.num_features = dim
        self.depth = depth
        #self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        self.out_indices= out_indices
        self.stem = nn.Sequential(
            nn.Conv2d(in_chans, dim, kernel_size=patch_size, stride=patch_size),
            activation(),
            nn.BatchNorm2d(dim)
        )
        self.outs_map = {
            0:0, 1:10, 2:20,3:30
        }
        self.depth= min(depth, max(self.outs_map.values())+1)
        self.blocks = nn.ModuleList(
            [nn.Sequential(
                    Residual(nn.Sequential(
                        nn.Conv2d(dim, dim, kernel_size, groups=dim, padding=(kernel_size-1)//2),
                        activation(),
                        nn.BatchNorm2d(dim)
                    )),
                    nn.Conv2d(dim, dim, kernel_size=1),
                    activation(),
                    nn.BatchNorm2d(dim)
            ) for i in range(self.depth)]
        )
        #self.pooling = nn.Sequential(
        #    nn.AdaptiveAvgPool2d((1, 1)),
        #    nn.Flatten()
        #)
        
        #self.deconv = nn.ConvTranspose2d(dim, 256, kernel_size, stride = patch_size, padding=0, output_padding = 0)
        self.apply(self._init_weights)
        self.init_weights(pretrained=pretrained)
        self.stem.eval()
        for param in self.stem.parameters():
            param.requires_grad = False
    def get_classifier(self):
        return self.head
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            #load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
            revise_keys = [(r'^module\.', '')]
            checkpoint = _load_checkpoint(pretrained, map_location='cpu', logger=logger)
            # OrderedDict is a subclass of dict
            if not isinstance(checkpoint, dict):
                raise RuntimeError(
                    f'No state_dict found in checkpoint file {pretrained}')
            # get state_dict from checkpoint
            if 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            elif 'model' in checkpoint: # for our model
                state_dict = checkpoint['model']
            else:
                state_dict = checkpoint
            # strip prefix of state_dict
            for p, r in revise_keys:
                state_dict = {re.sub(p, r, k): v for k, v in state_dict.items()}
            # load state_dict
            load_state_dict(self, state_dict, strict=False, logger=logger)

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
          
    def forward_features(self, x):
        x = self.stem(x)
        #save_tensor(x[0],'stem.png')
        outs = []
        for i in range(self.depth):
            x = self.blocks[i](x) 
            if i in self.outs_map.values():
                outs.append(x)
            #save_tensor(x[0],'tmp_{i}.png'.format(i=i))
        #x = self.blocks(x)
        #x = self.pooling(x)

        return outs
    
    def forward(self, x):
        outs = self.forward_features(x)

        #print('x ',x.shape)
        #x = self.head(x)
        #x = self.deconv(x)
        #print('x2',x.shape)

        s4 = F.interpolate(outs[0],scale_factor=1/4*7,mode='bilinear')
        s8 = F.interpolate(outs[1],scale_factor=1/8*7,mode='bilinear')
        s16 = F.interpolate(outs[2],scale_factor=1/16*7,mode='bilinear')
        s32 = F.interpolate(outs[3],scale_factor=1/32*7,mode='bilinear')

        new_outs = [s4,s8,s16,s32]
        #for i in range(len(outs)):
        #    print(new_outs[i].shape)
        #   save_tensor(new_outs[i][0],'tmp_{i}.png'.format(i=i))
        #exit()
        ret = []
        for indx in self.out_indices:
            ret.append(new_outs[indx])

        return ret



@BACKBONES.register_module()
class convmixer_1536_20(ConvMixer):
    def __init__(self, out_indices=[1,2,3],**kwargs):
        model_args = dict(dim=1536, depth=20, kernel_size=9, patch_size=7, **kwargs)
        super(ConvMixer, self).__init__(
            dim=model_args['dim'], depth=model_args['depth'], kernel_size=model_args['kernel_size'],patch_size=model_args['patch_size'],pretrained=kwargs['pretrained']
        )


@BACKBONES.register_module()
class convmixer_768_32(ConvMixer):
    def __init__(self, dim=None, out_indices=[1,2,3],**kwargs):
        model_args = dict(dim=768, depth=32, kernel_size=7, patch_size=7, activation=nn.ReLU, **kwargs)
        super(convmixer_768_32, self).__init__(
            dim=model_args['dim'], depth=model_args['depth'], kernel_size=model_args['kernel_size'],patch_size=model_args['patch_size'], activation=nn.ReLU,pretrained=kwargs['pretrained'],out_indices=out_indices
        )


#@register_model
#def convmixer_1024_20_ks9_p14(pretrained=False, **kwargs):
#    model_args = dict(dim=1024, depth=20, kernel_size=9, patch_size=14, **kwargs)
#    return _create_convmixer('convmixer_1024_20_ks9_p14', pretrained, **model_args)