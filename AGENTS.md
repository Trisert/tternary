# tternary

Ternary neural network with **two implementations** sharing the same dataset.

## Rust (burn-rs)

```bash
cargo build                                    # CPU (NdArray + OpenBLAS)
cargo run -- --steps 500 --epochs 10           # train
cargo run -- --tiny --steps 100 --epochs 3     # tiny config for quick testing
cargo run -- --generate 200                    # inference from checkpoints/best.safetensors
cargo run -r --features cuda -- --steps 500    # CUDA (requires Nix dev shell: nix develop .#cuda)
```

CLI: `--epochs`, `--steps`, `--lr`, `--generate`, `--small`, `--tiny`

Checkpoints: `checkpoints/*.safetensors` (burn-store format).

## Python/PyTorch

```bash
uv sync                              # install deps in .venv
uv run python -m pytternary.train --tiny --steps 100
uv run python -m pytternary.train --tiny --steps 100 --compile max-autotune
uv run python -m pytternary.train --hf-dataset   # load TinyStories via HF datasets
```

CLI: `--epochs`, `--steps`, `--lr`, `--generate`, `--small`, `--tiny`,
`--compile {default,reduce-overhead,max-autotune,max-autotune-no-cudagraphs}`,
`--hf-dataset`, `--ternary-threshold`, `--grad-clip`.

Checkpoints: `pytternary/checkpoints/*.pt`.

## Architecture

**TernaryTransformer** — token embed + position embed → N×BoltBlock → RMSNorm → output projection.

Each **BoltBlock**: residual `x + GatedConvMixer(norm(x))` + residual `x + GLUFFN(norm(x))`.

**TernaryLinear**: STE (straight-through estimator) — forward uses ternary weights `{-scale, 0, +scale}`, backward passes full-precision gradients. Threshold ratio configurable via `--ternary-threshold` (default 0.5, lower keeps more weights active).

**GatedConvMixer**: depthwise conv1d (left-only padding) gated by sigmoid(linear(x)).

**GLUFFN**: SiLU-gated FFN with ternary linear layers.

## Data

Shared between both implementations in `data/`:
- `tokenizer.json` — BPE tokenizer (vocab_size=4096), trained on TinyStories
- `tinystories_encoded_u16.bin` — pre-encoded token IDs as u16 LE (471M tokens, 940MB)

The Rust `main.rs` auto-downloads TinyStories and trains the tokenizer if missing. Python assumes pre-existing files.

**Python tokenizer gotcha**: the ByteLevel decoder is not serialized. Must set `tokenizer.decoder = decoders.ByteLevel()` after loading for clean space decoding.

## Nix

Dev shells via `flake.nix`:
- `nix develop` — CPU (OpenBLAS)
- `nix develop .#cuda` — CUDA (RTX 2060 / Tesla P100)

## .gitignore

Ignores: `/target`, `*.bin`, `data/tinystories_encoded.bin`, `pytternary/checkpoints/`,
`pytternary/src/pytternary.egg-info/`, `__pycache__/`, `*.pyc`.
