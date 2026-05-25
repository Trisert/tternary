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
    def __init__(self, tokenizer, max_seq_len: int, split: str = "train"):
        from datasets import load_dataset
        from tokenizers import Encoding

        dataset = load_dataset("roneneldan/TinyStories", split=split)
        self.max_seq_len = max_seq_len

        tokens = []
        for example in dataset:
            enc: Encoding = tokenizer.encode(example["text"])
            tokens.extend(enc.ids)

        self.tokens = np.array(tokens, dtype=np.int32)
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
