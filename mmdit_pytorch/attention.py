import torch
from torch import Tensor
import math

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from .normalization import MultiHeadRMSNorm


# utility funcs

def exists(val):
    return val is not None

def default(a, b):
    return a if exists(a) else b


# Self attention block (from navjak/base-transformer-NLP)

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
        key = self.w_k(k)
        value = self.w_v(v)

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



# Joint Attention block

class JointAttention(nn.Module):
    def __init__(self, dim_inputs: tuple[int, ...], dim_head = 64, num_heads = 8, qk_rmsnorm = False):
        
        super().__init__()

        self.dim_inputs = dim_inputs
        self.dim_head = dim_head
        self.num_heads = num_heads
        self.qk_rmsnorm = qk_rmsnorm

        dim_inner = dim_head * num_heads
        self.num_inputs = len(dim_inputs)

        self.proj_to_qkv = nn.ModuleList([nn.Linear(dim_input, dim_inner * 3, bias = False) for dim_input in dim_inputs]) # projection layer for all modalities

        self.attention = MultiHeadAttention(d_model = dim_inner, num_heads = num_heads, dropout = 0.1)

        self.proj_to_out = nn.ModuleList([nn.Linear(dim_inner, dim_input, bias=False) for dim_input in dim_inputs]) # output projection for all modalities
        
        self.q_rmsnorm = (None,) * self.num_inputs # default to tuples of "None" values if qk_rmsnorm is False
        self.k_rmsnorm = (None,) * self.num_inputs

        if qk_rmsnorm:
            self.q_rmsnorm = nn.ModuleList([MultiHeadRMSNorm(dim_head, heads = num_heads) for _ in range(num_inputs)])
            self.k_rmsnorm = nn.ModuleList([MultiHeadRMSNorm(dim_head, heads = num_heads) for _ in range(num_inputs)])
        
        self.register_buffer('dummy', torch.tensor(0), persistent = False)

    
    def forward(self, inputs: tuple[Tensor], masks:tuple[Tensor | None] | None = None):
        device = self.dummy.device
        assert len(inputs) == self.num_inputs

        masks = default(masks, (None,) * self.num_inputs)

        all_qkvs = []
        all_masks = []

        for x, mask, to_qkv, q_rmsnorm, k_rmsnorm in zip(inputs, masks, self.proj_to_qkv, self.q_rmsnorm, self.k_rmsnorm):

            qkv = to_qkv(x)
            
            # split into heads
            qkv = rearrange(qkv, 'b n (qkv h d) -> qkv b h n d', qkv=3, h=self.num_heads)

            # optional RMSNorm for q and k per modality
            if self.qk_rmsnorm:
                q, k, v = qkv
                q = q_rmsnorm(q)
                k = k_rmsnorm(k)
                qkv = torch.stack((q, k, v))

            all_qkvs.append(qkv)

            # mask for each modality - if not provided, assumes all valid
            if not exists(mask):
                mask = torch.ones(x.shape[:2], device = device, dtype = torch.bool)

            all_masks.append(mask)

        # combine all qkv and masks
        all_qkvs, packed_shape = pack(all_qkvs, 'qkv b h * d')
        all_masks, _ = pack(all_masks, 'b *')

        # attention
        q, k, v = all_qkvs.chunk(3, dim=0)  # split into q, k, v
        
        # reshape for MHA
        q = rearrange(q, 'b h n d -> b n (h d)')
        k = rearrange(k, 'b h n d -> b n (h d)')
        v = rearrange(v, 'b h n d -> b n (h d)')
        
        # attention
        output = self.attention(q, k, v, mask=all_masks)
        
        # separate by modality
        output = unpack(output, packed_shape, 'b * d')

        # output projection per modality
        all_outputs = []
        for out, proj_to_out in zip(output, self.proj_to_out):
            out = proj_to_out(out)
            all_outputs.append(out)

        return tuple(all_outputs)
    