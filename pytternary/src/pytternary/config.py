from dataclasses import dataclass, field


@dataclass
class AppConfig:
    vocab_size: int
    embed_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 6
    max_seq_len: int = 512
    batch_size: int = 16
    kernel_size: int = 8
    num_epochs: int = 10
    steps_per_epoch: int = 500
    learning_rate: float = 0.003
    ternary_threshold: float = 0.5
