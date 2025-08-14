from pathlib import Path

import numpy as np

from transformers import AutoTokenizer


class Dataloader:
    def __init__(self, batch_size, block_size):
        self.batch_size = batch_size
        self.block_size = block_size

        with open(Path().absolute() / "datasets" / "panchatantra-ryder-clean.txt") as f:
            text = f.read()
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            self.tokens = tokenizer.encode(text)

        print(f"Initialized dataloader with {len(self.tokens)} tokens")

    def __call__(self):
        tokens = self.tokens
        B = self.batch_size
        T = self.block_size
        D = len(tokens)
        num_sequences = (D - 1) // T
        num_batches = num_sequences // B

        for i in range(num_batches):
            x = []
            y = []
            for j in range(B):
                start_idx = (i * B + j) * T
                end_idx = start_idx + T
                x_seq = tokens[start_idx:end_idx]
                y_seq = tokens[start_idx + 1 : end_idx + 1]
                x.append(x_seq)
                y.append(y_seq)
            x = np.array(x)
            y = np.array(y)
            yield x, y


if __name__ == "__main__":
    dl = Dataloader(32, 128)
    it = dl()
    for x, y in it:
        print(x.shape, y.shape)
        assert x.shape == (32, 128)
