import random
import threading
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

        starts = [random.randrange(0, max_start) for _ in range(batch_size)]

        inputs = torch.empty(batch_size, seq, dtype=torch.long)
        targets = torch.empty(batch_size, seq, dtype=torch.long)

        for i, start in enumerate(starts):
            chunk = torch.from_numpy(self.data[start:start + seq + 1].copy()).long()
            inputs[i] = chunk[:seq]
            targets[i] = chunk[1:]

        return inputs, targets


class BackgroundPrefetcher:
    def __init__(self, dataset, batch_size: int, num_prefetch: int = 2):
        self.dataset = dataset
        self.batch_size = batch_size
        self.len = dataset.len
        self.queue = __import__("queue").Queue(maxsize=num_prefetch)
        self.worker = threading.Thread(target=self._run, daemon=True)
        self.worker.start()

    def _run(self):
        while True:
            self.queue.put(self.dataset.get_random_batch(self.batch_size))

    def get_random_batch(self, _batch_size):
        return self.queue.get()


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

        starts = [random.randrange(0, max_start) for _ in range(batch_size)]

        inputs = torch.empty(batch_size, seq, dtype=torch.long)
        targets = torch.empty(batch_size, seq, dtype=torch.long)

        for i, start in enumerate(starts):
            chunk = torch.from_numpy(self.tokens[start:start + seq + 1].copy()).long()
            inputs[i] = chunk[:seq]
            targets[i] = chunk[1:]

        return inputs, targets
