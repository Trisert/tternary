# tternary

Ternary neural network in **PyTorch**. Tokenizer implemented in Rust via PyO3.

## Setup

```bash
cd pytternary
uv sync                                   # install Python deps in .venv

# Build & install the Rust tokenizer
cd tokenizer_rs && maturin develop --uv && cd ..
```

## Training

```bash
uv run python -m pytternary.train --tiny --steps 100
```

See `AGENTS.md` for full CLI options.
