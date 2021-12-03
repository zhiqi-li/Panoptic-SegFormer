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

#=====

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


class SelfAttention(nn.Module):
    def __init__(self,cfg, dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x,key_padding_mask):
        B, N, C = x.shape
       
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if key_padding_mask is not None:
            #print('key_padding_mask0',key_padding_mask.shape)

            #key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,N,1)

            mask = torch.ones_like(attn)
            #print(attn.shape)
            for i,length in enumerate(key_padding_mask):
                #print(length)
                mask[i,:,:,:length] = 0
            assert mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
            #print('key_padding_mask1',mask.shape)

            

            attn.masked_fill_(mask.to(torch.bool), float("-inf"))
            del key_padding_mask
            del mask

        attn = attn.softmax(dim=-1)


        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        
        return x

class Attention(nn.Module):
    def __init__(self, cfg,dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.linear = nn.Sequential(
            nn.Linear(self.num_heads,self.num_heads),
            nn.ReLU(),
            nn.Linear(self.num_heads,1),
            nn.Sigmoid()            
        )
    def forward(self, query, key,value, key_padding_mask):
        B, N, C  = query.shape
        _, L, _ = key.shape
        #print('query, key, value', query.shape, value.shape, key.shape)
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3).contiguous()#.permute(2, 0, 3, 1, 4)
        k = self.k(key).reshape(B, L, self.num_heads, C // self.num_heads).permute(0,2,1,3).contiguous()   #.permute(2, 0, 3, 1, 4)
        
        v = self.v(value).reshape(B, L, self.num_heads, C // self.num_heads).permute(0,2,1,3).contiguous() #.permute(2, 0, 3, 1, 4)

        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,N,1)
            assert key_padding_mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
            #print('key_padding_mask1',key_padding_mask.shape)
            attn.masked_fill_(key_padding_mask, float("-inf"))
            del key_padding_mask
        mask = self.linear(attn.permute(0,2,3,1))  # B,NH, N,L
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).contiguous().reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x,mask
        
class AttentionTail(nn.Module):
    def __init__(self, cfg,dim, num_heads=2, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads =  num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.linear = nn.Sequential(
            nn.Linear(self.num_heads,self.num_heads),
            nn.ReLU(),
            nn.Linear(self.num_heads,1),
            nn.Sigmoid()            
        )
    
    def forward(self, query, key, key_padding_mask):
        B, N, C  = query.shape
        _, L, _ = key.shape
        #print('query, key, value', query.shape, value.shape, key.shape)
        q = self.q(query).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3).contiguous()#.permute(2, 0, 3, 1, 4)
        k = self.k(key).reshape(B, L, self.num_heads, C // self.num_heads).permute(0,2,1,3).contiguous()   #.permute(2, 0, 3, 1, 4)
        attn = (q @ k.transpose(-2, -1).contiguous()) * self.scale
        if key_padding_mask is not None:
            key_padding_mask = key_padding_mask.unsqueeze(1).unsqueeze(1).repeat(1,1,N,1)
            assert key_padding_mask.shape[-1] == attn.shape[-1], 'mask has incorrect dimensions'
            #print('key_padding_mask1',key_padding_mask.shape)
            attn.masked_fill_(key_padding_mask, float("-inf"))
            del key_padding_mask
        mask = self.linear(attn.permute(0,2,3,1))
        return mask 




class Block(nn.Module):

    def __init__(self, cfg,dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm):
        super().__init__()
        self.head_norm1 = norm_layer(dim)
        self.attn = Attention(cfg,
            dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.head_norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, query, key,value, key_padding_mask = None):
 
        x,mask = self.attn(query,key, value, key_padding_mask)
        query = query + self.drop_path(x)
        query = self.head_norm1(query)
     
        query = query + self.drop_path(self.mlp(query))
        query = self.head_norm2(query)
        return query,mask

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
class TransformerHead4(nn.Module):

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
        #dpr = [0 for _ in range(num_decoder_layers)]
        norm_layer = None
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = None
        act_layer = act_layer or nn.GELU
        block = Block(cfg,
                dim=d_model, num_heads=nhead, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=0, norm_layer=norm_layer, act_layer=act_layer)
        self.blocks = _get_clones(block, num_decoder_layers)
        self.attnen = AttentionTail(cfg,
            d_model, num_heads=nhead, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop_rate, proj_drop=0)

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
        
    def forward(self, memory, mask_memory,pos_memory, query_embed, mask_query,pos_query):
        if mask_memory is not None and isinstance(mask_memory,torch.Tensor):
            mask_memory = mask_memory.to(torch.bool)
        masks = []
        for block in self.blocks:
            query_embed,mask = block(self.with_pos_embed(query_embed, pos_query),self.with_pos_embed(memory, pos_memory),memory, key_padding_mask= mask_memory)
            masks.append(mask)
        attn = self.attnen(self.with_pos_embed(query_embed, pos_query),self.with_pos_embed(memory, pos_memory), key_padding_mask = mask_memory)
        return attn,masks