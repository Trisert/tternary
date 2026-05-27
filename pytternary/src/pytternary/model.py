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
            BoltBlock(config.embed_dim, config.hidden_dim, config.kernel_size, config.max_seq_len, config.ternary_threshold)
            for _ in range(config.num_layers)
        ])
        self.norm = TernaryRMSNorm(config.embed_dim)

        self.output_head = nn.Linear(config.embed_dim, config.vocab_size, bias=False)
        init_scale = math.sqrt(2.0 / config.embed_dim)
        nn.init.uniform_(self.output_head.weight, -init_scale, init_scale)

        self.register_buffer("_pos_buffer", torch.arange(config.max_seq_len), persistent=False)
        self._step = 0

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        batch, seq = input_ids.shape
        positions = self._pos_buffer[:seq].unsqueeze(0).expand(batch, -1)
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return x

    def forward_logits(self, input_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.forward(input_ids)
        return self.output_head(hidden)

    def forward_training(self, input_ids: torch.Tensor, targets: torch.Tensor) -> dict:
        logits = self.forward_logits(input_ids)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return {"loss": loss, "logits": logits}

    @torch.no_grad()
    def forward_step(self, input_ids: torch.Tensor) -> torch.Tensor:
        positions = self._pos_buffer[self._step:self._step + 1].unsqueeze(0)
        self._step += 1
        x = self.token_embed(input_ids) + self.pos_embed(positions)
        for block in self.blocks:
            x = block.forward_step(x)
        x = self.norm(x)
        return self.output_head(x)

    def reset_cache(self):
        for block in self.blocks:
            block.reset_cache()
        self._step = 0

    def quantize_weights(self):
        for block in self.blocks:
            block.mixer.gate.quantize_weights()
            block.ffn.w_gate.quantize_weights()
            block.ffn.w_up.quantize_weights()
            block.ffn.w_down.quantize_weights()

    def dequantize_weights(self):
        for block in self.blocks:
            block.mixer.gate.dequantize_weights()
            block.ffn.w_gate.dequantize_weights()
            block.ffn.w_up.dequantize_weights()
            block.ffn.w_down.dequantize_weights()

    @torch.no_grad()
    def generate(self, tokenizer, num_tokens: int, temperature: float = 0.8, device: str = "cpu"):
        self.eval()
        self.reset_cache()
        self.quantize_weights()
        current_tokens = [2]
        vocab_size = self.config.vocab_size
        inp = torch.tensor([[2]], dtype=torch.long, device=device)
        for _ in range(num_tokens):
            logits = self.forward_step(inp)
            last_logits = logits[0, -1, :vocab_size] / temperature
            probs = F.softmax(last_logits, dim=-1)
            sampled = torch.multinomial(probs, 1).item()
            current_tokens.append(sampled)
            inp = torch.tensor([[sampled]], dtype=torch.long, device=device)
        self.dequantize_weights()
        text = tokenizer.decode(current_tokens, True)
        return text[:500]

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
