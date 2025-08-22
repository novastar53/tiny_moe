from pathlib import Path
from typing import Callable, List
from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np

import tiktoken
from transformers import AutoTokenizer

from logging_config import setup_logging

logger = setup_logging()


class Dataloader:
    def __init__(self, batch_size, block_size):
        self.batch_size = batch_size
        self.block_size = block_size

        with open(Path().absolute() / "datasets" / "panchatantra-ryder-clean.txt") as f:
            text = f.read()
            #tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            tokenizer = tiktoken.get_encoding("gpt2")
            self.tokens = tokenizer.encode(text)

        logger.info(f"Initialized dataloader with {len(self.tokens)} tokens")

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
        logger.info(f"Batch shapes - x: {x.shape}, y: {y.shape}")
        assert x.shape == (32, 128)


class BaseDataLoader(ABC):
    """
    Abstract base class for data loaders that yield token batches.
    Subclasses must implement _list_shards and _load_shard.
    """

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        device_rank: int,
        label: str | None = None,
        quiet: bool = False,
        start_shard: int = 0,
        start_shard_pos: int = 0,
    ):
        # Common initialization
        self.enc = tiktoken.get_encoding("gpt2")
        self.eot = self.enc._special_tokens["<|endoftext|>"]
        self.B = batch_size
        self.T = block_size
        self.D = device_rank
        self.label = label

        # List and filter shards
        self.shards = self._list_shards(label)
        self.cur_shard = start_shard
        self.shard_pos = start_shard_pos
        self.shard = self._load_shard()
        self.shard_size = len(self.shard)

        if not quiet:
            logger.info(f"""{self.__class__.__name__} initialized:
------------------------
label:          {label}
shards:         {len(self.shards):,}
shard size:     {self.shard_size:,}
batch size:     {self.B}
block size:     {self.T}
device rank:    {self.D}
start shard:    {start_shard}
start pos:      {start_shard_pos}
------------------------""")

    def __len__(self):
        return len(self.shards) * self.shard_size

    def __call__(self):
        # preallocate buffer
        buf_size = self.B * self.T * self.D + 1
        buf = np.zeros((buf_size,), dtype=np.uint16)

        if self.shard_pos + buf_size < self.shard_size:
            buf[:] = self.shard[self.shard_pos : self.shard_pos + buf_size]
            self.shard_pos += buf_size
        else:
            # fill the remaining shard
            buf_prefix = self.shard_size - self.shard_pos
            buf[:buf_prefix] = self.shard[self.shard_pos :]
            buf_pos = buf_prefix

            # load the next shard
            self.cur_shard += 1
            self.shard = self._load_shard()
            self.shard_pos = 0

            # fill full shards
            while buf_pos + self.shard_size <= buf_size:
                buf[buf_pos : buf_pos + self.shard_size] = self.shard
                buf_pos += self.shard_size
                self.cur_shard += 1
                self.shard = self._load_shard()

            # final partial shard
            self.shard_pos = buf_size - buf_pos
            buf[buf_pos:] = self.shard[: self.shard_pos]

        X = buf[:-1].reshape((self.D, self.B, self.T))
        Y = buf[1:].reshape((self.D, self.B, self.T))
        return X, Y

    @abstractmethod
    def _list_shards(self, label: str | None) -> list[str]:
        """Return list of shard identifiers, filtered by label."""
        pass

    @abstractmethod
    def _load_shard(self) -> np.ndarray:
        """Load and return current shard as a 1D numpy array."""
        pass



class CloudDataLoader(BaseDataLoader):
    """
    DataLoader that reads token shards from a Google Cloud Storage bucket.
    """

    def __init__(
        self,
        bucket_name: str,
        bucket_prefix: str,
        batch_size: int,
        block_size: int,
        device_rank: int,
        label: str | None = None,
        quiet: bool = False,
        start_shard: int = 0,
        start_shard_pos: int = 0,
    ):
        from google.cloud import storage

        self.bucket_name = bucket_name
        self.bucket_prefix = bucket_prefix
        self.client = storage.Client()
        self.bucket = self.client.bucket(bucket_name)
        super().__init__(
            batch_size,
            block_size,
            device_rank,
            label,
            quiet,
            start_shard=start_shard,
            start_shard_pos=start_shard_pos,
        )

    def _list_shards(self, label):
        blobs = self.bucket.list_blobs(prefix=self.bucket_prefix)
        return [
            blob.name for blob in blobs if (label is None or label in blob.name)
        ]

    def _load_shard(self):
        from io import BytesIO

        if self.cur_shard >= len(self.shards):
            self.cur_shard = 0
        shard_name = self.shards[self.cur_shard]
        blob = self.bucket.blob(shard_name)
        data = blob.download_as_bytes()
        tokens = np.load(BytesIO(data))
        if not isinstance(tokens, np.ndarray):
            tokens = tokens["arr_0"]
        self.shard_size = len(tokens)
        return tokens



class BlendedCloudDataLoader():
    def __init__(
        self,
        batch_size: int,
        block_size: int,
        bucket_names: List[str],
        bucket_prefixes: List[str],
        proportions: List[float],
        device_rank: int, 
        label: str | None = None,
        quiet: bool = False,
        start_shards: List[int] = None,
        start_shard_positions: List[int] = None,
    ):
        self.B = batch_size
        self.T = block_size
        self.D = device_rank
        batch_sizes = self._calc_batch_sizes(batch_size, proportions)
        logger.info(f"Initializing blended dataset with batch sizes: {batch_sizes}")
        self.dataloaders = []
        n = len(bucket_names)
        # Default to zeros if not provided
        if start_shards is None:
            start_shards = [0] * n
        if start_shard_positions is None:
            start_shard_positions = [0] * n
        for i, (bucket_name, bucket_prefix, batch_size) in enumerate(zip(bucket_names, bucket_prefixes, batch_sizes)):
            self.dataloaders.append(
                CloudDataLoader(
                    bucket_name,
                    bucket_prefix,
                    batch_size,
                    block_size,
                    device_rank,
                    label,
                    quiet,
                    start_shard=start_shards[i],
                    start_shard_pos=start_shard_positions[i],
                )
            )

    def __call__(self):
        X, Y = [], []
        for dataloader in self.dataloaders:
            _X, _Y = dataloader()
            X.append(_X)
            Y.append(_Y)
        
        X = np.concatenate(X, axis=1)
        Y = np.concatenate(Y, axis=1)

        assert(X.shape == (self.D, self.B, self.T))

        return X, Y

    def _calc_batch_sizes(self, B, ratios):
        ratios = np.array(ratios)
        assert(B >= len(ratios))
        batch_proportions = jnp.astype(B * ratios / sum(ratios), jnp.int32)
        if sum(batch_proportions) < B:
            diff = B - sum(batch_proportions)
            while diff > 0:
                min_idx = jnp.argmin(batch_proportions)
                batch_proportions = batch_proportions.at[min_idx].add(1)
                diff -= 1
        assert(sum(batch_proportions) == B)
        return batch_proportions
