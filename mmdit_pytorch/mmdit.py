
import torch
from torch import nn
from torch import Tensor
from torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from typing import Optional, Tuple

from hyper_connections import HyperConnections, Residual

# add import line for joint Attention after i create a separate attention .py file

# create a fodler called attention and put the attention files there
# it will have 2 files - MHA and Joint Attention

# utility funcs

def exists(a):
    return a is not None

def default(a, b):
    return a if exists(a) else b

def softclamp(t, val):
    return (t/val).tanh() * val


# MultiHead RMSNorm

class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads=1):
        super().__init__()
        self.scale = dim ** 0.5 # for scaling the output
        self.gamma = nn.Parameter(torch.ones(heads, dim)) # gamma parameter



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

        


#####################################
#####################################
# Self attention block
#####################################
# Edit this for mmdit
#####################################
#####################################

class MultiHeadAttention(nn.Module):
    def __init__(self, df_model: int, num_heads: int, dropout: float):
        super().__init__()     
        self.num_heads = num_heads
        self.d_model = d_model   

        assert d_model % num_heads == 0, "d_model must be divisible by num_heads"
        
        self.d_k = d_model // num_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    @staticmethod # allows calling this function without an instance of the class
    def attention(query, key, value, mask, dropout: nn.Dropout):
        d_k = query.shape[-1]   
        attn_scores =  (query @ key.transpose(-2, -1)) / math.sqrt(d_k)

        # apply mask (for Masked Multi-Head Attention in the decoder)
        if mask is not None:
            attn_scores.masked_fill(mask == 0, float('-inf'))

        # softmax
        attn_scores = attn_scores.softmax(dim=-1)  # (batch_size, num_heads, seq_len, seq_len)
        
        # dropout
        if dropout is not None:
            attn_scores = dropout(attn_scores)

        return (attn_scores @ value), attn_scores


    def forward(self, q, k, v, mask=None):
        query = self.w_q(q) # (batch_size, seq_len, d_model)
        key = self.w_k(k) # (batch_size, seq_len, d_model)
        value = self.w_v(v) # (batch_size, seq_len, d_model)

        # # (batch_size, seq_len, d_model ) --> (batch_size, num_heads, seq_len, d_k )
        query = query.view(query.shape[0], query.shape[1], self.num_heads, self.d_k).transpose(1, 2) # (batch_size, num_heads, seq_len, d_k )
        key = key.view(key.shape[0], key.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(value.shape[0], value.shape[1], self.num_heads, self.d_k).transpose(1, 2)
        
        att_x, attn_scores = MultiHeadAttention.attention(query, key, value, mask, self.dropout) # att_x shape (batch_size, num_heads, seq_len, d_k )

        x = x.transpose(1,2).contiguous().view(x.shape[0], -1,self.num_heads * self.d_model) # (batch_size, seq_len, d_model)

        x = self.w_o(x) # (batch_size, seq_len, d_model)
        return x
    
#####################################
#####################################





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

        self.attention =    # write my own attention class --> args: (q, k, v, mask = all_masks)



