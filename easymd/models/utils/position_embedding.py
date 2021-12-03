import torch
import torch.nn as nn
import math

def xyposEmbd(tensor=None,shape=None):
    assert tensor is not None or shape is not None
    if tensor is None:
        b, h, w = shape
    else:
        b, c, h, w = tensor.shape
    x_range = torch.linspace(-1, 1, w, device=tensor.device)
    y_range = torch.linspace(-1, 1, h, device=tensor.device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([b, 1, -1, -1])
    x = x.expand([b, 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)
    return coord_feat

def xyposEmbd2(tensor=None,shape=None):
    assert tensor is not None or shape is not None
    if tensor is None:
        b, h, w = shape
    else:
        b, c, h, w = tensor.shape
    x_range = torch.linspace(0, 1, w, device=tensor.device)
    y_range = torch.linspace(0, 1, h, device=tensor.device)
    y, x = torch.meshgrid(y_range, x_range)
    y = y.expand([b, 1, -1, -1])
    x = x.expand([b, 1, -1, -1])
    coord_feat = torch.cat([x, y], 1)
    return coord_feat

class RelPositionEmbedding(nn.Module):
    def __init__(self, num_pos_feats=64, pos_norm=True):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.fc = nn.Linear(4, self.num_pos_feats,bias=False)
        #nn.init.orthogonal_(self.fc.weight)
        #self.fc.weight.requires_grad = False
        self.pos_norm = pos_norm
        if self.pos_norm:
            self.norm = nn.LayerNorm(self.num_pos_feats)
    def forward(self, tensor):
        #mask = nesttensor.mask
        B,C,H,W = tensor.shape
        #print('tensor.shape',  tensor.shape)
        y_range = (torch.arange(H) / float(H - 1)).to(tensor.device)
        #y_axis = torch.stack((y_range, 1-y_range),dim=1)
        y_axis = torch.stack((torch.cos(y_range * math.pi), torch.sin(y_range * math.pi)), dim=1)
        y_axis = y_axis.reshape(H, 1, 2).repeat(1, W, 1).reshape(H * W, 2)

        x_range = (torch.arange(W) / float(W - 1)).to(tensor.device)
        #x_axis =torch.stack((x_range,1-x_range),dim=1)
        x_axis = torch.stack((torch.cos(x_range * math.pi), torch.sin(x_range * math.pi)), dim=1)
        x_axis = x_axis.reshape(1, W, 2).repeat(H, 1, 1).reshape(H * W, 2)
        x_pos = torch.cat((y_axis, x_axis), dim=1)
        x_pos = self.fc(x_pos)
        #for i in range(50,51):
        #print(i)
        #print('xpos,', x_pos.max(),x_pos.min())
        if self.pos_norm:
            x_pos = self.norm(x_pos)
        #print('xpos,', x_pos.max(),x_pos.min())
        return x_pos