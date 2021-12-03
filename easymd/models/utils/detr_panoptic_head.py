# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
DETR Transformer class.

Copy-paste from torch.nn.Transformer with modifications:
    * positional encodings are passed in MHattention
    * extra LN at the end of encoder is removed
    * decoder returns a stack of activations from all decoding layers
"""
import copy
from typing import Optional, List

from numpy.ma.core import dot, masked_values

import torch
import torch.nn.functional as F
from torch import nn, Tensor
from functools import partial
from mmdet.models.utils.builder import TRANSFORMER
import math
#=====
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from easymd.models.utils.visual import save_tensor


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x



        
class AttentionTail(nn.Module):
    def __init__(self, cfg,dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
    
        self._reset_parameters()

    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def forward(self, query, key, key_padding_mask,hw_lvl=None):
        B, N, C  = query.shape
        _, L, _ = key.shape
        #print('query, key, value', query.shape, value.shape, key.shape)
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3).contiguous()#.permute(2, 0, 3, 1, 4)
        k = self.k(key).reshape(B, L, self.num_heads, C // self.num_heads).permute(0,2,1,3).contiguous()   #.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        
        
        #attn = attn
        attn = attn.permute(0,2,3,1)
        #print('attn',attn.shape,hw_lvl)
        wedge_1 = hw_lvl[0][0]*hw_lvl[0][1]
        wedge_2 = wedge_1+ hw_lvl[1][0]*hw_lvl[1][1]
        
        wedge_3 = hw_lvl[2][0]*hw_lvl[2][1]
        #print(wedge_1,wedge_2)
        feats_l1 = attn[:,:,:wedge_1,:]
        feats_l2 = attn[:,:,wedge_1:wedge_2,:]
        feats_l3 = attn[:,:,wedge_2:,:]
       
        x = key[:,wedge_2:,:]
        #am = feats_l3.permute(0,1,3,2)
        #am = am.reshape(-1,wedge_3)
        #am_max = am.max(-1)[0][...,None]
        #am_min = am.min(-1)[0][...,None]
        #am = (am-am_min)/(am_max-am_min)
        #am = am.reshape(-1,*hw_lvl[2]) 
        
        #save_tensor(am[1:2],'d-detr-ms-1.png',convert=True)
        #save_tensor(am[9:10],'d-detr-ms-10.png',convert=True)
        attn_map = feats_l3.permute(0,1,3,2).reshape(B,-1,8,*hw_lvl[2]) 

        x = x.permute(0,2,1).reshape(B,256,*hw_lvl[2])
        
        return x, attn_map




def drop_path(x, drop_prob: float = 0., training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output

def _get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)

@TRANSFORMER.register_module()
class TransformerHead15(nn.Module):

    def __init__(self,cfg=None, d_model=16, nhead=2, num_encoder_layers=6,
                 num_decoder_layers=1, dim_feedforward=64, dropout=0.1,
                 activation="relu", normalize_before=False,
                 return_intermediate_dec=False):
        super().__init__()

        #encoder_layer = TransformerEncoderLayer(d_model, nhead, dim_feedforward,
                                               # dropout, activation, normalize_before)
        #encoder_norm = nn.LayerNorm(d_model) if normalize_before else None
        #self.encoder = TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)
        mlp_ratio = 4
        qkv_bias =True
        qk_scale = None
        drop_rate = 0
        attn_drop_rate = 0

        self.attnen = AttentionTail(cfg,
            d_model, num_heads=nhead, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=0)
        fpn_dims = [256,256,256]
        self.panoptic_head = MaskHeadSmallConv(256+8,fpn_dims,256)
        self._reset_parameters()

    
    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
    def with_pos_embed(self, tensor, pos: Optional[Tensor]):
        if pos is None:
            return tensor
        else:
            #print('tensor',tensor.shape, pos.shape)
            #print('pos',pos.max(),pos.min())

            return tensor+pos
        #return tensor if pos is None else tensor + pos
        
    def forward(self, memory, mask_memory,pos_memory, query_embed, mask_query,pos_query,hw_lvl,fpns):
        if mask_memory is not None and isinstance(mask_memory,torch.Tensor):
            mask_memory = mask_memory.to(torch.bool)

        x, attn = self.attnen(self.with_pos_embed(query_embed, pos_query),self.with_pos_embed(memory, pos_memory), key_padding_mask = mask_memory,hw_lvl=hw_lvl)
    
        masks = self.panoptic_head(x,attn,fpns).permute(1,0,2,3)
      
        return masks




def _expand(tensor, length: int):
    return tensor.unsqueeze(1).repeat(1, int(length), 1, 1, 1).flatten(0, 1)

class MaskHeadSmallConv(nn.Module):
    """
    Simple convolutional head, using group norm.
    Upsampling is done using a FPN approach
    """

    def __init__(self, dim, fpn_dims, context_dim):
        super().__init__()

        inter_dims = [dim, context_dim // 2, context_dim // 4, context_dim // 8, context_dim // 16, context_dim // 64]
        self.lay1 = torch.nn.Conv2d(dim, dim, 3, padding=1)
        self.gn1 = torch.nn.GroupNorm(8, dim)
        self.lay2 = torch.nn.Conv2d(dim, inter_dims[1], 3, padding=1)
        self.gn2 = torch.nn.GroupNorm(8, inter_dims[1])
        self.lay3 = torch.nn.Conv2d(inter_dims[1], inter_dims[2], 3, padding=1)
        self.gn3 = torch.nn.GroupNorm(8, inter_dims[2])
        self.lay4 = torch.nn.Conv2d(inter_dims[2], inter_dims[3], 3, padding=1)
        self.gn4 = torch.nn.GroupNorm(8, inter_dims[3])
        self.lay5 = torch.nn.Conv2d(inter_dims[3], inter_dims[4], 3, padding=1)
        self.gn5 = torch.nn.GroupNorm(8, inter_dims[4])
        self.out_lay = torch.nn.Conv2d(inter_dims[4], 1, 3, padding=1)

        self.dim = dim

        self.adapter1 = torch.nn.Conv2d(fpn_dims[0], inter_dims[1], 1)
        self.adapter2 = torch.nn.Conv2d(fpn_dims[1], inter_dims[2], 1)
        self.adapter3 = torch.nn.Conv2d(fpn_dims[2], inter_dims[3], 1)
        self.sigmoid = torch.nn.Sigmoid()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: Tensor, bbox_mask: Tensor, fpns: List[Tensor]):
     
        x = torch.cat([_expand(x, bbox_mask.shape[1]), bbox_mask.flatten(0, 1)], 1)
     
        x = self.lay1(x)
        x = self.gn1(x)
        x = F.relu(x)
        x = self.lay2(x)
        x = self.gn2(x)
        x = F.relu(x)

        #for each in fpns:
        #    print(each.shape)
        fpns = [fpns[2],fpns[1],fpns[0]]
        #print('x',x.shape, fpns[0].shape)
        cur_fpn = self.adapter1(fpns[0])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay3(x)
        x = self.gn3(x)
        x = F.relu(x)
     
        cur_fpn = self.adapter2(fpns[1])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay4(x)
        x = self.gn4(x)
        x = F.relu(x)

        cur_fpn = self.adapter3(fpns[2])
        if cur_fpn.size(0) != x.size(0):
            cur_fpn = _expand(cur_fpn, x.size(0) // cur_fpn.size(0))
        x = cur_fpn + F.interpolate(x, size=cur_fpn.shape[-2:], mode="nearest")
        x = self.lay5(x)
        x = self.gn5(x)
        x = F.relu(x)
        x =self.sigmoid(self.out_lay(x))
        #save_tensor(x[0:1],'d-detr-ms-00.png',convert=True)
        #save_tensor(x[1:2],'d-detr-ms-01.png',convert=True)
        #print ('x', x.shape)
        return x