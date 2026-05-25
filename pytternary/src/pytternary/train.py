import argparse
import math
import os
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
from tqdm import tqdm

from .config import AppConfig
from .model import TernaryTransformer
from .dataset import EncodedDataset, HFCausalLMDataset


_ROOT = Path(__file__).resolve().parent.parent.parent.parent
ENCODED_FILE = str(_ROOT / "data" / "tinystories_encoded_u16.bin")
TOKENIZER_FILE = str(_ROOT / "data" / "tokenizer.json")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--lr", type=float, default=0.003)
    parser.add_argument("--generate", type=int, default=0)
    parser.add_argument("--small", action="store_true")
    parser.add_argument("--tiny", action="store_true")
    parser.add_argument("--compile", choices=("default", "reduce-overhead", "max-autotune", "max-autotune-no-cudagraphs"), default=None)
    parser.add_argument("--hf-dataset", action="store_true", help="Load TinyStories via HF datasets instead of mmap")
    parser.add_argument("--ternary-threshold", type=float, default=0.5, help="Ternary quantization threshold ratio (default 0.5, lower = more weights kept active)")
    parser.add_argument("--grad-clip", type=float, default=0.0, help="Max gradient norm for clipping (0 = no clipping)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"=== Ternary Transformer (PyTorch / {device.upper()}) ===\n")

    from tokenizers import Tokenizer, decoders
    tokenizer = Tokenizer.from_file(TOKENIZER_FILE)
    tokenizer.decoder = decoders.ByteLevel()
    vocab_size = tokenizer.get_vocab_size()
    print(f"Vocabulary size: {vocab_size}")

    config = AppConfig(vocab_size)
    if args.tiny:
        config.embed_dim = 24
        config.hidden_dim = 128
        config.num_layers = 5
        config.kernel_size = 5
    elif args.small:
        config.embed_dim = 192
        config.hidden_dim = 384
        config.num_layers = 4
    config.num_epochs = args.epochs
    config.steps_per_epoch = args.steps
    config.learning_rate = args.lr
    config.ternary_threshold = args.ternary_threshold

    tag = " (tiny)" if args.tiny else " (small)" if args.small else ""
    print(f"Config{tag}: embed_dim={config.embed_dim}, hidden={config.hidden_dim}, "
          f"layers={config.num_layers}, max_seq={config.max_seq_len}, kernel={config.kernel_size}")

    model = TernaryTransformer(config).to(device)
    compute_loss = model.forward_training
    if args.compile:
        print(f"Compiling with mode={args.compile} ...")
        compute_loss = torch.compile(compute_loss, mode=args.compile, fullgraph=True)
    print(f"Parameters: {model.num_parameters():,}")

    if args.hf_dataset:
        print("Loading TinyStories via HF datasets (tokenizing on-the-fly)...")
        dataset = HFCausalLMDataset(tokenizer, config.max_seq_len)
    else:
        dataset = EncodedDataset(ENCODED_FILE, config.max_seq_len)
    num_tokens = dataset.len
    print(f"Dataset: {num_tokens:,} tokens ({num_tokens * 2 / 1e6:.1f} MB on disk)")

    optim = AdamW(model.parameters(), lr=config.learning_rate, betas=(0.9, 0.999))
    warmup_epochs = 1
    min_lr = config.learning_rate * 0.1

    best_loss = float("inf")
    ckpt_dir = Path("checkpoints")
    ckpt_dir.mkdir(exist_ok=True)

    clip_val = args.grad_clip
    if clip_val:
        print(f"Gradient clipping: max_norm={clip_val}")
    if args.ternary_threshold != 0.5:
        print(f"Ternary threshold: {args.ternary_threshold}")

    print(f"\nTraining for {config.num_epochs} epochs, {config.steps_per_epoch} steps/epoch, "
          f"lr={config.learning_rate}\n")

    total_start = time.time()

    for epoch in range(config.num_epochs):
        epoch_start = time.time()
        epoch_loss = 0.0

        if epoch < warmup_epochs:
            current_lr = min_lr + (config.learning_rate - min_lr) * (epoch + 1) / warmup_epochs
        elif config.num_epochs <= warmup_epochs:
            current_lr = config.learning_rate
        else:
            progress = (epoch - warmup_epochs) / max(config.num_epochs - warmup_epochs, 1)
            current_lr = min_lr + 0.5 * (config.learning_rate - min_lr) * (1.0 + math.cos(math.pi * progress))

        for param_group in optim.param_groups:
            param_group["lr"] = current_lr

        model.train()
        for _ in tqdm(range(config.steps_per_epoch), desc=f"Epoch {epoch + 1:>3}"):
            inputs, targets = dataset.get_random_batch(config.batch_size)
            inputs, targets = inputs.to(device), targets.to(device)

            optim.zero_grad()
            loss = compute_loss(inputs, targets)["loss"]
            loss.backward()
            if clip_val:
                torch.nn.utils.clip_grad_norm_(model.parameters(), clip_val)
            optim.step()
            epoch_loss += loss.item()

        avg_loss = epoch_loss / config.steps_per_epoch
        elapsed = time.time() - epoch_start
        print(f"Epoch {epoch + 1:>3} | Loss: {avg_loss:.4f} | LR: {current_lr:.6f} | Time: {elapsed:.2f}s")

        ckpt_path = ckpt_dir / f"epoch_{epoch + 1:04d}.pt"
        torch.save(model.state_dict(), ckpt_path)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), ckpt_dir / "best.pt")
            print(f"  New best loss: {best_loss:.4f}")

        if epoch % 5 == 0 or epoch == config.num_epochs - 1:
            text = model.generate(tokenizer, 60, device=device)
            print(f"  {text}")

    total_time = time.time() - total_start
    print(f"\nTotal training time: {total_time:.2f}s")
    print(f"Best loss: {best_loss:.4f}")

    best_path = ckpt_dir / "best.pt"
    if best_path.exists():
        print("\n--- Best Checkpoint Generated Text ---")
        model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
        text = model.generate(tokenizer, 200, device=device)
        print(text)
    else:
        print("\n--- Final Model Generated Text ---")
        text = model.generate(tokenizer, 200, device=device)
        print(text)


if __name__ == "__main__":
    main()
