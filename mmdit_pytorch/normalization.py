import torch
from torch import nn


# MultiHead RMSNorm
class MultiHeadRMSNorm(Module):
    def __init__(self, dim, heads = 1):
        super().__init__()
        self.scale = dim ** 0.5 # for scaling the output
        self.gamma = nn.Parameter(torch.ones(heads, dim)) # gamma parameter

    def forward(self, x):
        return F.normalize(x, dim = -1) * self.gamma * self.scale



# LayerNorm





# adaLN