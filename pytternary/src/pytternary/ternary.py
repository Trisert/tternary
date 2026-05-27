import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TernaryLinear(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, ternary_threshold: float = 0.5):
        super().__init__()
        init_scale = math.sqrt(2.0 / in_dim)
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim).uniform_(-init_scale, init_scale))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        self.ternary_threshold = ternary_threshold
        self._quantized_weight = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = self.weight
        if self._quantized_weight is not None:
            return F.linear(x, self._quantized_weight, self.bias)
        w_detached = w.detach()
        w_abs = w_detached.abs()
        scale = w_abs.mean()
        w_ternary = torch.where(
            w_abs > scale * self.ternary_threshold,
            w_detached.sign() * scale,
            torch.zeros_like(w_detached),
        )
        w_ste = w_ternary + (w - w_detached)
        return F.linear(x, w_ste, self.bias)

    @torch.no_grad()
    def quantize_weights(self):
        w = self.weight
        w_abs = w.abs()
        scale = w_abs.mean()
        self._quantized_weight = torch.where(
            w_abs > scale * self.ternary_threshold,
            w.sign() * scale,
            torch.zeros_like(w),
        )

    def dequantize_weights(self):
        self._quantized_weight = None
