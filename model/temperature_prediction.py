
import math
import pdb

import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable

class LogitsTemPredNet(nn.Module):
    def __init__(self,):
        super(LogitsTemPredNet, self).__init__()
        #self.t_net1 = MLPMixer(41)
        #self.t_net2 = MLPMixer(41)
        #self.t_net3 = MLPMixer(41)
        self.t_net1 = nn.Linear(40,1)
        self.t_net2 = nn.Linear(40,1)
        self.t_net3 = nn.Linear(40,1)
        self.relu =nn.ReLU()
    def forward(self, r1, r2, r3):
        #data shape N, 256, 16, 25
        #r1_t = r1.mean(-1) # N, 256,16
        #r2_t = r2.mean(-1) # N, 256, 16
        #r3_t = r3.mean(-1) # N, 256, 16

        #r1_s = r1.mean(-2) # N, 256,25
        #r2_s = r2.mean(-2) # N, 256, 25
        #r3_s = r3.mean(-2) # N, 256, 25
        #r1 = torch.cat([r1_t, r1_s], -1) #N, 256, 41
        #r2 = torch.cat([r2_t, r2_s], -1)
        #r3 = torch.cat([r3_t, r3_s], -1)
        #t1 = self.t_net1(r1)
        #t2 = self.t_net2(r2)
        #t3 = self.t_net3(r3)
        #return torch.nn.functional.sigmoid(t1), torch.nn.functional.sigmoid(t2), torch.nn.functional.sigmoid(t3)
        t1 = self.t_net1(r1)
        t2 = self.t_net2(r2)
        t3 = self.t-net3(r3)
        return self.relu(t1), self.relu(t2), self.relu(t3)

from torch import nn
from functools import partial
from einops.layers.torch import Rearrange, Reduce

pair = lambda x: x if isinstance(x, tuple) else (x, x)

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(tokens, out_channels=1, dim=256, depth=1,expansion_factor = 4, expansion_factor_token = 0.5, dropout = 0.):
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear
    return nn.Sequential(
        nn.Linear(tokens, dim),
        *[nn.Sequential(
            PreNormResidual(dim, FeedForward(dim, expansion_factor, dropout, chan_first)),
            PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last))
        ) for _ in range(depth)],
        nn.LayerNorm(dim),
        Reduce('b n c -> b c', 'mean'),
        nn.Linear(dim, out_channels)
    )