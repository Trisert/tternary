import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import AppConfig
from .modules import TernaryEmbedding, TernaryRMSNorm, BoltBlock


class TernaryTransformer(nn.Module):
    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.token_embed = TernaryEmbedding(config.vocab_size, config.embed_dim)
        self.pos_embed = TernaryEmbedding(config.max_seq_len, config.embed_dim)
        self.blocks = nn.ModuleList([
            BoltBlock(config.embed_dim, config.hidden_dim, config.kernel_size, config.max_seq_len)
            for _ in range(config.num_layers)
        ])
        self.norm = TernaryRMSNorm(config.embed_dim)

        init_scale = math.sqrt(2.0 / config.embed_dim)
        self.output_weight = nn.Parameter(
            torch.empty(config.embed_dim, config.vocab_size).uniform_(-init_scale, init_scale)
        )

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        device = input_ids.device

        positions = torch.arange(seq, device=device).unsqueeze(0).expand(batch, -1)

        x = self.token_embed(input_ids) + self.pos_embed(positions)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        return x

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.forward(input_ids)
        return hidden @ self.output_weight

    def forward_training(self, input_ids: torch.Tensor, targets: torch.Tensor) -> dict:
        logits = self.forward_logits(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def generate(self, tokenizer, num_tokens: int, temperature: float = 0.8, device: str = "cpu"):
        self.eval()
        current_tokens = [2]
        vocab_size = self.config.vocab_size

        for _ in range(num_tokens):
            inp = torch.tensor([current_tokens], dtype=torch.long, device=device)
            logits = self.forward_logits(inp)
            last_logits = logits[0, -1, :vocab_size] / temperature
            probs = F.softmax(last_logits, dim=-1)
            sampled = torch.multinomial(probs, 1).item()
            current_tokens.append(sampled)

        text = tokenizer.decode(current_tokens, True)
        return text[:500]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
