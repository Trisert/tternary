import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .ternary import TernaryLinear


class TernaryEmbedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int):
        super().__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.embedding(x)


class TernaryRMSNorm(nn.Module):
    def __init__(self, embed_dim: int, eps: float = 1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(embed_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = (x.pow(2).mean(-1, keepdim=True) + self.eps).sqrt()
        return x * rms.reciprocal() * self.weight


class TernaryDepthwiseConv(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int, _seq_len: int):
        super().__init__()
        init_scale = math.sqrt(2.0 / kernel_size)
        self.kernel_size = kernel_size
        self.conv1d = nn.Conv1d(
            embed_dim, embed_dim, kernel_size,
            groups=embed_dim, bias=False, padding=0
        )
        nn.init.uniform_(self.conv1d.weight, -init_scale, init_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)
        x_t = F.pad(x_t, (self.kernel_size - 1, 0))
        out = self.conv1d(x_t)
        return out.transpose(1, 2)


class GatedConvMixer(nn.Module):
    def __init__(self, embed_dim: int, kernel_size: int, seq_len: int):
        super().__init__()
        self.conv = TernaryDepthwiseConv(embed_dim, kernel_size, seq_len)
        self.gate = TernaryLinear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        conv_out = self.conv(x)
        gate_raw = self.gate(x)
        sig = torch.sigmoid(gate_raw)
        return conv_out * sig


class TernaryGLUFFN(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.w_gate = TernaryLinear(embed_dim, hidden_dim)
        self.w_up = TernaryLinear(embed_dim, hidden_dim)
        self.w_down = TernaryLinear(hidden_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)


class BoltBlock(nn.Module):
    def __init__(self, embed_dim: int, hidden_dim: int, kernel_size: int, seq_len: int):
        super().__init__()
        self.norm1 = TernaryRMSNorm(embed_dim)
        self.mixer = GatedConvMixer(embed_dim, kernel_size, seq_len)
        self.norm2 = TernaryRMSNorm(embed_dim)
        self.ffn = TernaryGLUFFN(embed_dim, hidden_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mixer(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x
