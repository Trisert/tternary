import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        init_scale = math.sqrt(2.0 / in_dim)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-init_scale, init_scale))
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        w_detached = w.detach()

        scale = w_detached.abs().mean()
        threshold = scale * 0.5

        pos = (w_detached > threshold).float()
        neg = (w_detached < -threshold).float()
        w_ternary = (pos - neg) * scale

        w_ste = w_ternary + (w - w_detached)

        return F.linear(x, w_ste, self.bias)
