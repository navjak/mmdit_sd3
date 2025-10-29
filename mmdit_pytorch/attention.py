# Self attention block


import torch
from torch import nn
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, dropout: float = 0.1):
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
    
    def forward(self, q, k, v, mask = None):
        
        batch_size, seq_len, _ = q.size()
        
        query = self.w_q(q) # (batch_size, seq_len, d_model)
        key = self.w_q(k)
        value = self.w_q(v)

        # (batch_size, seq_len, d_model ) ==> (batch_size, num_heads, seq_len, d_k )
        query = query.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) # [batch_size, num_heads, seq_len, d_k]
        key = key.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        value = value.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1, 2) 

        # calculate scaled dot-product attention
        attn_scores = (query @ key.transpose(-2, -1)) / math.sqrt(self.d_k)  # [batch_size, num_heads, seq_len, seq_len]
        
        # apply mask 
        if mask is not None: 
            attn_scores = attn_scores.masked_fill(mask == 0, float('-inf'))
        
        attn = torch.softmax(attn_scores, dim=-1)
        attn = self.dropout(attn)
        
        x = (attn @ value)  # [batch_size, num_heads, seq_len, d_k]
        
        # concatenate the heads
        x = x.transpose(1, 2).contiguous().view(batch_size, seq_len, self.num_heads * self.d_k) # (batch_size, seq_len, d_model)
        x = self.w_o(x)  # final linear -- (batch_size, seq_len, d_model)
        return x




class JointAttention(nn.Module):