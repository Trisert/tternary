# tternary

Ternary neural network in **PyTorch**. Tokenizer implemented in Rust via PyO3.

## Setup

```bash
cd pytternary
uv sync                                   # install Python deps in .venv
uv pip install maturin                     # install maturin for Rust tokenizer build

# Build & install the Rust tokenizer
uv run maturin develop --manifest-path tokenizer_rs/Cargo.toml
```

## Training

```bash
uv run python -m pytternary.train --tiny --steps 100
uv run python -m pytternary.train --tiny --steps 100 --compile max-autotune
uv run python -m pytternary.train --hf-dataset   # load TinyStories via HF datasets
uv run python -m pytternary.prepare_code                  # download + tokenize code dataset
uv run python -m pytternary.train --dataset code --steps 500 --tiny  # train on code
```

CLI: `--epochs`, `--steps`, `--lr`, `--generate`, `--small`, `--tiny`,
`--compile {default,reduce-overhead,max-autotune,max-autotune-no-cudagraphs}`,
`--hf-dataset`, `--dataset {tinystories,code}`, `--ternary-threshold`, `--grad-clip`.

Checkpoints: `pytternary/checkpoints/*.pt`.

## Tokenizer (Rust / PyO3)

The tokenizer lives in `pytternary/tokenizer_rs/` and is built with [maturin](https://www.maturin.rs/).

```bash
cd pytternary/tokenizer_rs
maturin develop      # build + install in .venv
```

Python API:

```python
from pytternary_tokenizer import Tokenizer, train_tokenizer, tokenize_dataset_file

tok = Tokenizer.load("data/tokenizer.json")
ids = tok.encode("Hello world")               # list[int]

train_tokenizer("input.txt", "tok.json", 4096)       # train BPE + ByteLevel
tokenize_dataset_file("in.txt", "tok.json", "out.bin")  # parallel encode to u16
```

**Tip**: use `prepare_code` to download and tokenize the code dataset, then train:

```bash
uv run python -m pytternary.prepare_code
uv run python -m pytternary.train --dataset code --steps 500
```

## Data

In `data/`:
- `tokenizer.json` — BPE tokenizer (vocab_size=4096), trained on TinyStories
- `tokenizer_code.json` — BPE tokenizer (vocab_size=4096), trained on code snippets
- `tinystories_encoded_u16.bin` — pre-encoded token IDs as u16 LE (471M tokens, 940MB)
- `code_encoded_u16.bin` — pre-encoded code token IDs as u16 LE

## Architecture

**TernaryTransformer** — token embed + position embed → N×BoltBlock → RMSNorm → output projection.

Each **BoltBlock**: residual `x + GatedConvMixer(norm(x))` + residual `x + GLUFFN(norm(x))`.

**TernaryLinear**: STE (straight-through estimator) — forward uses ternary weights `{-scale, 0, +scale}`, backward passes full-precision gradients. Threshold ratio configurable via `--ternary-threshold` (default 0.5, lower keeps more weights active).

**GatedConvMixer**: depthwise conv1d (left-only padding) gated by sigmoid(linear(x)).

**GLUFFN**: SiLU-gated FFN with ternary linear layers.
