import random
import numpy as np
import torch


class EncodedDataset:
    def __init__(self, path: str, max_seq_len: int):
        self.data = np.memmap(path, dtype=np.uint16, mode="r")
        self.len = len(self.data)
        self.max_seq_len = max_seq_len

    def get_random_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = max(0, self.len - self.max_seq_len - 1)
        seq = self.max_seq_len

        input_buf = []
        target_buf = []

        for _ in range(batch_size):
            start = random.randrange(0, max_start)
            chunk = torch.from_numpy(self.data[start:start + seq + 1].copy()).long()
            input_buf.append(chunk[:seq])
            target_buf.append(chunk[1:])

        return torch.stack(input_buf), torch.stack(target_buf)


class HFCausalLMDataset:
    def __init__(self, tokenizer, max_seq_len: int, split: str = "train", cache_file: str | None = None):
        import os
        from tqdm import tqdm

        self.max_seq_len = max_seq_len

        if cache_file is not None and os.path.exists(cache_file):
            print(f"Loading cached tokenized dataset from {cache_file}")
            self.tokens = np.load(cache_file)
        else:
            from datasets import load_dataset
            dataset = load_dataset("roneneldan/TinyStories", split=split)
            texts = [example["text"] for example in dataset]

            tokens = []
            for text in tqdm(texts, desc="Tokenizing"):
                tokens.extend(tokenizer.encode(text))

            self.tokens = np.array(tokens, dtype=np.int32)

            if cache_file is not None:
                np.save(cache_file, self.tokens)
                print(f"Saved cached tokenized dataset to {cache_file}")

        self.len = len(self.tokens)

    def get_random_batch(self, batch_size: int) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = max(0, self.len - self.max_seq_len - 1)
        seq = self.max_seq_len

        input_buf = []
        target_buf = []

        for _ in range(batch_size):
            start = random.randrange(0, max_start)
            chunk = torch.from_numpy(self.tokens[start:start + seq + 1].copy()).long()
            input_buf.append(chunk[:seq])
            target_buf.append(chunk[1:])

        return torch.stack(input_buf), torch.stack(target_buf)
