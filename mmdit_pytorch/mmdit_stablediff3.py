import torch
from torch import nn, Tensor
import torch.nn.functional as F

from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange

from typing import Optional, Tuple

from hyper_connections import HyperConnections, Residual

from .attention import JointAttention
from .normalization import MultiHeadRMSNorm, AdaLN


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


# MMDIT block

class MMDiTLayer(nn.Module):
    def __init__(
        self,
        dim_text: int,
        dim_image: int,
        dim_cond: int | None = None,
        dim_head: int = 64,
        heads: int = 8,
        qk_rmsnorm: bool = False,
        flash_attn: bool = False,
        num_residual_streams: int = 1,
        mlp_kwargs: dict = dict()
    ):
        super().__init__()

        # residual / hyper connections
        residual_type = Residual if num_residual_streams == 1 else HyperConnections

        self.text_attn_residual_fn = residual_type(num_residual_streams, dim=dim_text)
        self.text_mlp_residual_fn = residual_type(num_residual_streams, dim=dim_text)

        self.img_attn_residual_fn = residual_type(num_residual_streams, dim=dim_image)
        self.img_mlp_residual_fn = residual_type(num_residual_streams, dim=dim_image)

        # handle optional time conditioning
        has_cond = exists(dim_cond)
        self.has_cond = has_cond

        if has_cond:
            dim_gammas = (*((dim_text,) * 4), *((dim_image,) * 4))
            dim_betas = (*((dim_text,) * 2),*((dim_image,) * 2))

            self.cond_dims = (*dim_gammas, *dim_betas)

            to_cond_linear = nn.Linear(dim_cond, sum(self.cond_dims))

            self.to_cond = nn.Sequential(
                Rearrange('b d -> b 1 d'),
                nn.SiLU(),
                to_cond_linear
            )

            # zero init weights and biases
            nn.init.zeros_(to_cond_linear.weight)
            nn.init.zeros_(to_cond_linear.bias)
            nn.init.constant_(to_cond_linear.bias[:sum(dim_gammas)], 1.) # set gamma biases to 1

        # adaLN for both modalities (img and text)
        self.text_attn_norm = AdaLN(dim_text, dim_cond, num_modulations=2) if has_cond else nn.LayerNorm(dim_text)
        self.img_attn_norm = AdaLN(dim_image, dim_cond, num_modulations=2) if has_cond else nn.LayerNorm(dim_image)

        self.text_mlp_norm = AdaLN(dim_text, dim_cond, num_modulations=2) if has_cond else nn.LayerNorm(dim_text)
        self.img_mlp_norm = AdaLN(dim_image, dim_cond, num_modulations=2) if has_cond else nn.LayerNorm(dim_image)

        # joint attention
        self.joint_attn = JointAttention(
            dim_inputs=(dim_text, dim_image),
            dim_head=dim_head,
            num_heads=heads,
            qk_rmsnorm=qk_rmsnorm,
            flash_attn=flash_attn
        )

        # mlp layers
        self.text_mlp = MLP(dim_text, **mlp_kwargs)
        self.img_mlp = MLP(dim_image, **mlp_kwargs)

    def forward(
        self,
        *,
        text_tokens: Tensor,
        image_tokens: Tensor,
        text_mask: Optional[Tensor] = None,
        time_cond: Optional[Tensor] = None,
        skip_text_mlp: bool = True
    ) -> Tuple[Tensor, Tensor]:
        
        assert not (exists(time_cond) ^ self.has_cond), 'time condition must be passed in if dim_cond is set at init. it should not be passed in if not set'

        if self.has_cond:
            (
                text_pre_attn_gamma,
                text_post_attn_gamma,
                text_pre_mlp_gamma,
                text_post_mlp_gamma,
                img_pre_attn_gamma,
                img_post_attn_gamma,
                img_pre_mlp_gamma,
                img_post_mlp_gamma,
                text_pre_attn_beta,
                text_pre_mlp_beta,
                img_pre_attn_beta,
                img_pre_mlp_beta
                ) = self.to_cond(time_cond).split(self.cond_dims, dim = -1)

        # handle adaLN
        text_tokens, add_text_residual = self.text_attn_residual_fn(text_tokens)
        img_tokens, add_img_residual = self.img_attn_residual_fn(image_tokens)

        text_tokens = self.text_attn_norm(text_tokens, time_cond) if self.has_cond else self.text_attn_norm(text_tokens)
        img_tokens = self.img_attn_norm(img_tokens, time_cond) if self.has_cond else self.img_attn_norm(img_tokens)

        # apply pre-attention conditioning
        if self.has_cond:
            text_tokens = text_tokens * text_pre_attn_gamma + text_pre_attn_beta
            img_tokens = img_tokens * img_pre_attn_gamma + img_pre_attn_beta

        # apply joint attention
        text_tokens, img_tokens = self.joint_attn(
            inputs=(text_tokens, img_tokens),
            masks=(text_mask, None)
        )

        # condition attention output
        if self.has_cond:
            text_tokens = text_tokens * text_post_attn_gamma
            img_tokens = img_tokens * img_post_attn_gamma

        # add attention residual
        text_tokens = add_text_residual(text_tokens)
        img_tokens = add_img_residual(img_tokens)

        # handle mlp layer adaLN
        if not skip_text_mlp:
            text_tokens, add_text_residual = self.text_mlp_residual_fn(text_tokens)
            text_tokens = self.text_mlp_norm(text_tokens, time_cond) if self.has_cond else self.text_mlp_norm(text_tokens)

        img_tokens, add_img_residual = self.img_mlp_residual_fn(img_tokens)
        img_tokens = self.img_mlp_norm(img_tokens, time_cond) if self.has_cond else self.img_mlp_norm(img_tokens)

        # apply mlp conditioning
        if self.has_cond:
            img_tokens = img_tokens * img_pre_mlp_gamma + img_pre_mlp_beta

        # img mlp layer
        img_tokens = self.img_mlp(img_tokens)
        
        # apply post mlp conditioning
        if self.has_cond:
            img_tokens = img_tokens * img_post_mlp_gamma

        img_tokens = add_img_residual(img_tokens)

        if skip_text_mlp:
            return text_tokens, img_tokens

        # text mlp layer
        text_tokens = self.text_mlp(text_tokens)
        
        # apply post mlp conditioning
        if self.has_cond:
            text_tokens = text_tokens * text_post_mlp_gamma
            
        text_tokens = add_text_residual(text_tokens)

        return text_tokens, img_tokens



# Main MMDiT model 

class MMDiT(nn.Module):
    def __init__(
        self,
        *,
        depth: int,
        dim_image: int,
        num_register_tokens: int = 0,
        final_norm: bool = True,
        num_residual_streams: int = 4,
        **mmdit_kwargs
    ):
        super().__init__()

        self.expand_streams, self.reduce_streams = HyperConnections.get_expand_reduce_stream_functions(num_residual_streams, disable = num_residual_streams == 1)

        # optional register tokens
        self.has_register_tokens = num_register_tokens >0
        self.register_tokens = nn.Parameter(torch.zeros(num_register_tokens, dim_image))
        nn.init.normal_(self.register_tokens, std =0.02)

        # stacked mmdit layers
        self.layers = nn.ModuleList([
            MMDiTLayer(
                dim_image = dim_image,
                num_residual_streams=  num_residual_streams,
                **mmdit_kwargs
            ) for _ in range(depth)
        ])

        # final norm
        self.norm = MultiHeadRMSNorm(dim_image) if final_norm else nn.Identity()

    def forward(
        self,
        *,
        text_tokens: Tensor,
        img_tokens: Tensor,
        text_mask: Optional[Tensor] = None,
        time_cond: Optional[Tensor] = None,
        should_skip_last_mlp: bool = True
    ) -> Tuple[Tensor, Tensor]:

        # handle optional register tokens
        if self.has_register_tokens:
            register_tokens = repeat(self.register_tokens, 'n d -> b n d', b= img_tokens.shape[0])
            img_tokens, packed_shape = pack([register_tokens, img_tokens], 'b * d')

        # expand streams for residuals
        text_tokens = self.expand_streams(text_tokens)
        img_tokens = self.expand_streams(img_tokens)

        # process through layers
        for ind, layer in enumerate(self.layers):
            is_last = ind == (len(self.layers) - 1)

            text_tokens, img_tokens = layer(
                time_cond = time_cond,
                text_tokens = text_tokens,
                image_tokens = img_tokens,
                text_mask = text_mask,
                skip_text_mlp = is_last and should_skip_last_mlp
            )

        # unpack register tokens if used
        if self.has_register_tokens:
            _, img_tokens = unpack(img_tokens, packed_shape, 'b * d')

        # reduce streams and normalize
        text_tokens = self.reduce_streams(text_tokens)
        img_tokens = self.reduce_streams(img_tokens)

        img_tokens = self.norm(img_tokens)

        return text_tokens, img_tokens