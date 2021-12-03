import copy
import math
import warnings
from mmdet.models.utils.builder import TRANSFORMER

import torch
import torch.nn as nn

from mmcv import ConfigDict
from mmcv.cnn import (Linear, build_activation_layer, build_norm_layer,
                      constant_init, xavier_init)
from mmcv.ops.multi_scale_deform_attn import (
    MultiScaleDeformableAttnFunction, multi_scale_deformable_attn_pytorch)
from mmcv.runner.base_module import BaseModule
from mmcv.utils import build_from_cfg
from mmcv.cnn.bricks.registry import (ATTENTION, POSITIONAL_ENCODING, TRANSFORMER_LAYER,
                       TRANSFORMER_LAYER_SEQUENCE)

from mmcv.cnn.bricks.transformer import build_attention, build_transformer_layer_sequence



class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class FFN_w_trick(BaseModule):
    """Implements feed-forward networks (FFNs) with residual connection.

    Args:
        embed_dims (int): The feature dimension. Same as
            `MultiheadAttention`.
        feedforward_channels (int): The hidden dimension of FFNs.
        num_fcs (int, optional): The number of fully-connected layers in
            FFNs. Default: 2.
        act_cfg (dict, optional): The activation config for FFNs.
            Default: dict(type='ReLU')
        dropout (float, optional): Probability of an element to be
            zeroed. Default 0..
        add_residual (bool, optional): Whether to add the
            residual connection. Default: `True`.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 embed_dims,
                 feedforward_channels,
                 num_fcs=2,
                 act_cfg=dict(type='ReLU', inplace=True),
                 dropout=0.,
                 add_residual=True,
                 init_cfg=None):
        super(FFN_w_trick, self).__init__(init_cfg)
        assert num_fcs >= 2, 'num_fcs should be no less ' \
            f'than 2. got {num_fcs}.'
        self.embed_dims = embed_dims
        self.feedforward_channels = feedforward_channels
        self.num_fcs = num_fcs
        self.act_cfg = act_cfg
        self.dropout = dropout
        activate = build_activation_layer(act_cfg)

        
        in_channels = embed_dims
        

        self.fc1 = Linear(in_channels, feedforward_channels)
        self.dwconv = DWConv(feedforward_channels)
        self.ac1 =  activate
        self.drp = nn.Dropout(dropout)
        self.fc2 = Linear(feedforward_channels, embed_dims)
        self.dropout = nn.Dropout(dropout)
        self.add_residual = add_residual

    def forward(self, x, residual=None,hw_lvl=None,**kwargs):
        """Forward function for `FFN`.

        The function would add x to the output tensor if residue is None.
        """
        #print(hw_lvl)
        y=x
        s8,s16,s32,s64 = hw_lvl[0],hw_lvl[1],hw_lvl[2],hw_lvl[3]
        
        I1 = s8[0]*s8[1]
        I2 = s16[0]*s16[1]+I1
        I3 = s32[0]*s32[1]+I2
        x = x.permute(1,0,2)
        x = self.fc1(x)
        x1 = self.dwconv(x[:,:I1],*s8)
        x2 = self.dwconv(x[:,I1:I2],*s16)
        x3 = self.dwconv(x[:,I2:I3],*s32)
        x4 = self.dwconv(x[:,I3:],*s64)
        x = torch.cat([x1,x2,x3,x4],1)
        x = self.ac1(x)
        x = self.drp(x)
        x = self.fc2(x)
        x = x.permute(1,0,2)
        if not self.add_residual:
            return self.dropout(x)
            
        if residual is None:
            residual = y
        return residual + self.dropout(x)


@TRANSFORMER_LAYER.register_module()
class BaseTransformerLayer_w_trick(BaseModule):
    """Base `TransformerLayer` for vision transformer.

    It can be built from `mmcv.ConfigDict` and support more flexible
    customization, for example, using any number of `FFN or LN ` and
    use different kinds of `attention` by specifying a list of `ConfigDict`
    named `attn_cfgs`. It is worth mentioning that it supports `prenorm`
    when you specifying `norm` as the first element of `operation_order`.
    More details about the `prenorm`: `On Layer Normalization in the
    Transformer Architecture <https://arxiv.org/abs/2002.04745>`_ .

    Args:
        attn_cfgs (list[`mmcv.ConfigDict`] | obj:`mmcv.ConfigDict` | None )):
            Configs for `self_attention` or `cross_attention` modules,
            The order of the configs in the list should be consistent with
            corresponding attentions in operation_order.
            If it is a dict, all of the attention modules in operation_order
            will be built with this config. Default: None.
        feedforward_channels (int): The hidden dimension for FFNs.
            Default: None.
        ffn_dropout (float): Probability of an element to be zeroed
            in ffn. Default 0..
        operation_order (tuple[str]): The execution order of operation
            in transformer. Such as ('self_attn', 'norm', 'ffn', 'norm').
            Support `prenorm` when you specifying first element as `norm`.
            Default：None.
        act_cfg (dict): The activation config for FFNs.
            Default: dict(type='ReLU')
        norm_cfg (dict): Config dict for normalization layer.
            Default: dict(type='LN').
        ffn_num_fcs (int): The number of fully-connected layers in FFNs.
            Default：2.
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Default: None.
    """

    def __init__(self,
                 attn_cfgs=None,
                 feedforward_channels=None,
                 ffn_dropout=0.,
                 operation_order=None,
                 act_cfg=dict(type='ReLU', inplace=True),
                 norm_cfg=dict(type='LN'),
                 ffn_num_fcs=2,
                 init_cfg=None):

        super(BaseTransformerLayer_w_trick, self).__init__(init_cfg)
        assert set(operation_order) & set(
            ['self_attn', 'norm', 'ffn', 'cross_attn']) == \
            set(operation_order), f'The operation_order of' \
            f' {self.__class__.__name__} should ' \
            f'contains all four operation type ' \
            f"{['self_attn', 'norm', 'ffn', 'cross_attn']}"
        num_attn = operation_order.count('self_attn') + operation_order.count(
            'cross_attn')
        if isinstance(attn_cfgs, ConfigDict):
            attn_cfgs = [copy.deepcopy(attn_cfgs) for _ in range(num_attn)]
        else:
            assert num_attn == len(attn_cfgs), f'The length ' \
                f'of attn_cfg {num_attn} is ' \
                f'not consistent with the number of attention' \
                f'in operation_order {operation_order}.'
        self.init_cfg = init_cfg
        self.num_attn = num_attn
        self.feedforward_channels = feedforward_channels
        self.ffn_dropout = ffn_dropout
        self.operation_order = operation_order
        self.act_cfg = act_cfg
        self.norm_cfg = norm_cfg
        self.ffn_num_fcs = ffn_num_fcs
        self.pre_norm = operation_order[0] == 'norm'
        self.attentions = nn.ModuleList()

        index = 0
        for operation in operation_order:
            if operation in ['self_attn', 'cross_attn']:
                attention = build_attention(attn_cfgs[index])
                self.attentions.append(attention)
                index += 1

        self.embed_dims = self.attentions[0].embed_dims
        self.ffns = nn.ModuleList()
        num_ffns = operation_order.count('ffn')
        for _ in range(num_ffns):
            self.ffns.append(
                FFN_w_trick(self.embed_dims, feedforward_channels, ffn_num_fcs,
                    act_cfg, ffn_dropout))

        self.norms = nn.ModuleList()
        num_norms = operation_order.count('norm')
        for _ in range(num_norms):
            self.norms.append(build_norm_layer(norm_cfg, self.embed_dims)[1])

    def forward(self,
                query,
                key,
                value,
                query_pos=None,
                key_pos=None,
                attn_masks=None,
                query_key_padding_mask=None,
                key_padding_mask=None,
                **kwargs):
        """Forward function for `TransformerDecoderLayer`.

        **kwargs contains some specific arguments of attentions.

        Args:
            query (Tensor): Input query with the shape
                `(num_queries, bs, embed_dims)`.
            key (Tensor): The key tensor with shape
                `(num_keys, bs, embed_dims)`.
            value (Tensor): The value tensor with shape
                `(num_keys, bs, embed_dims)`.
            query_pos (Tensor): The positional encoding for `query`.
                Default: None.
            key_pos (Tensor): The positional encoding for `key`.
                Default: None.
            attn_masks (List[Tensor] | None): 2D Tensor used in
                calculation of corresponding attention. The length of
                it should equal to the number of `attention` in
                `operation_order`. Default: None.
            query_key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_queries]. Only used in `self_attn` layer.
                Defaults to None.
            key_padding_mask (Tensor): ByteTensor for `query`, with
                shape [bs, num_keys]. Default: None.

        Returns:
            Tensor: forwarded results with shape [num_queries, bs, embed_dims].
        """

        norm_index = 0
        attn_index = 0
        ffn_index = 0
        inp_residual = query
        if attn_masks is None:
            attn_masks = [None for _ in range(self.num_attn)]
        elif isinstance(attn_masks, torch.Tensor):
            attn_masks = [
                copy.deepcopy(attn_masks) for _ in range(self.num_attn)
            ]
            warnings.warn(f'Use same attn_mask in all attentions in '
                          f'{self.__class__.__name__} ')
        else:
            assert len(attn_masks) == self.num_attn, f'The length of ' \
                        f'attn_masks {len(attn_masks)} must be equal ' \
                        f'to the number of attention in ' \
                        f'operation_order {self.num_attn}'

        for layer in self.operation_order:
            if layer == 'self_attn':
                temp_key = temp_value = query
                query = self.attentions[attn_index](
                    query,
                    temp_key,
                    temp_value,
                    inp_residual if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=query_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=query_key_padding_mask,
                    **kwargs)
                attn_index += 1
                inp_residual = query

            elif layer == 'norm':
                query = self.norms[norm_index](query)
                norm_index += 1

            elif layer == 'cross_attn':
                query = self.attentions[attn_index](
                    query,
                    key,
                    value,
                    inp_residual if self.pre_norm else None,
                    query_pos=query_pos,
                    key_pos=key_pos,
                    attn_mask=attn_masks[attn_index],
                    key_padding_mask=key_padding_mask,
                    **kwargs)
                attn_index += 1
                inp_residual = query

            elif layer == 'ffn':
                query = self.ffns[ffn_index](
                    query, inp_residual if self.pre_norm else None,**kwargs)
                ffn_index += 1

        return query


@TRANSFORMER.register_module()
class Transformer_o_encoder(BaseModule):
    """Implements the DETR transformer.

    Following the official DETR implementation, this module copy-paste
    from torch.nn.Transformer with modifications:

        * positional encodings are passed in MultiheadAttention
        * extra LN at the end of encoder is removed
        * decoder returns a stack of activations from all decoding layers

    See `paper: End-to-End Object Detection with Transformers
    <https://arxiv.org/pdf/2005.12872>`_ for details.

    Args:
        encoder (`mmcv.ConfigDict` | Dict): Config of
            TransformerEncoder. Defaults to None.
        decoder ((`mmcv.ConfigDict` | Dict)): Config of
            TransformerDecoder. Defaults to None
        init_cfg (obj:`mmcv.ConfigDict`): The Config for initialization.
            Defaults to None.
    """

    def __init__(self, encoder=None, decoder=None, init_cfg=None):
        super(Transformer_o_encoder, self).__init__(init_cfg=init_cfg)
        #assert encoder==None

        self.encoder = build_transformer_layer_sequence(encoder)
        #self.decoder = build_transformer_layer_sequence(decoder)
        self.embed_dims = self.encoder.embed_dims


    def init_weights(self):
        # follow the official DETR to init parameters
        for m in self.modules():
            if hasattr(m, 'weight') and m.weight.dim() > 1:
                xavier_init(m, distribution='uniform')
        self._is_init = True

    def forward(self, x, mask, query_embed, pos_embed):
        assert False, 'error transformer_o_decoder'
        return None