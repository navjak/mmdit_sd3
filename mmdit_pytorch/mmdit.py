
import torch
from torch import nn
from torch import Tensor
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from typing import Optional, Tuple

from hyper_connections import HyperConnections, Residual

# import attention.py and norm.py here


# utility funcs

def exists(a):
    return a is not None

def default(a, b):
    return a if exists(a) else b

def softclamp(t, val):
    return (t/val).tanh() * val



# MLP block (from navjak/base-transformer-NLP)

class MLP(nn.Module):
    def __init__(self, dim: int, mult: int = 4, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim * mult)
        self.act = nn.GELU()
        self.dropout1 = nn.Dropout(dropout)
        self.fc2 = nn.Linear(dim * mult, dim)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout1(x)
        x = self.fc2(x)
        x = self.dropout2(x)
        return x


'''    
# Modulated Joint attention block

class JointAttention(nn.Module):
    def __init(self, 
               dim_inputs:tuple[int],
               dim_head = 64,
               num_heads = 64,
               qk_rmsnorm = True,
               flash = False,
               softclamp = False, # used to avoid instability
                softclamp_val = 50,
                attn_kwargs: dict = dict() # remove maybe
    ):
        super().__init__()
        # b - vatch, h - heads, n - sequence, d - feature dim
        dim_inner = dim_heads * num_heads
        num_inputs = len(dim_inputs)
        self.num_inputs = num_inputs

        self.to_qkv = nn.ModuleList([])
        
        self.split_heads = Rearrange('b n (qkv h d) -> qkv b h n d', h = heads, qkv = 3)

        # self.attention =    # write my own attention class --> args: (q, k, v, mask = all_masks)
'''

