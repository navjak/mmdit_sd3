import torch
from torch import nn
import torch.nn.functional as F

# MultiHead RMSNorm
class MultiHeadRMSNorm(nn.Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5 # for scaling the output
        self.gamma = nn.Parameter(torch.ones(heads, dim)) # gamma parameter

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale
    

# adaLN
class AdaLN(nn.Module):
    def __init__(self, dim, cond_dim, num_modulations=1, eps=1e-6):
        """
        - cond_dim: conditioning vector dimension (pooled text + timestep embedding)
        - num_modulations: number of (gamma, beta) pairs required (set to 1 for default; set > 1 if several modulations from one cond vector)
        """
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False, eps=eps)
        self.modulation = nn.Linear(cond_dim, num_modulations * dim * 2)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)
        self.num_modulations = num_modulations

    def forward(self, x, cond, modulation_index=0):
        """
        - x: [B, N, dim] or [B, dim]
        - cond: [B, cond_dim]
        - modulation_index: which (gamma, beta) pair to use (0 by default)
        """
        x_norm = self.norm(x)
        mod_params = self.modulation(cond)  # [batch_size, num_modulations * dim * 2]
        params = mod_params.view(x.shape[0], self.num_modulations, 2, x_norm.shape[-1])
        gamma = params[:, modulation_index, 0, :]  # [batch_size, dim]
        beta = params[:, modulation_index, 1, :]   # [batch_size, dim]
        # broadcast
        while gamma.dim() < x_norm.dim():
            gamma = gamma.unsqueeze(1)
            beta = beta.unsqueeze(1)
        return gamma * x_norm + beta
