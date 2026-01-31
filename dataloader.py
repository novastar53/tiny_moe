import os
from pathlib import Path
from typing import List
from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np

import tiktoken

from logging_config import setup_logging

logger = setup_logging()

# Magic number for modded-nanogpt .bin format
BIN_MAGIC_NUMBER = 20240520
BIN_HEADER_SIZE = 256  # 256 int32 values = 1024 bytes


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
            logger.info(
                f"""{self.__class__.__name__} initialized:
------------------------
label:          {label}
shards:         {len(self.shards):,}
shard size:     {self.shard_size:,}
batch size:     {self.B}
block size:     {self.T}
device rank:    {self.D}
start shard:    {start_shard}
start pos:      {start_shard_pos}
------------------------"""
            )

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


class DataLoader(BaseDataLoader):
    def __init__(
        self,
        dirpath: str,
        batch_size: int,
        block_size: int,
        device_rank: int,
        label: str | None = None,
        quiet: bool = False,
        start_shard: int = 0,
        start_shard_pos: int = 0,
    ):
        self.dirpath = os.path.abspath(dirpath)
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
        if not os.path.exists(self.dirpath):
            raise FileNotFoundError(f"Directory not found: {self.dirpath}")
        shards = os.listdir(self.dirpath)
        logger.info(f"Found {len(shards)} files in {self.dirpath}")
        if label is not None:
            shards = [s for s in shards if label in s]
            logger.info(f"After filtering with label '{label}': {len(shards)} shards")
        return shards

    def _load_shard(self):
        if self.cur_shard >= len(self.shards):
            self.cur_shard = 0
        shard = self.shards[self.cur_shard]
        shard_path = os.path.join(self.dirpath, shard)

        if shard.endswith(".bin"):
            tokens = load_bin_shard(shard_path)
        else:
            tokens = np.load(shard_path)
            if not isinstance(tokens, np.ndarray):
                tokens = tokens["arr_0"]

        self.shard_size = len(tokens)
        return tokens


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
        return [blob.name for blob in blobs if (label is None or label in blob.name)]

    def _load_shard(self):
        from io import BytesIO

        if self.cur_shard >= len(self.shards):
            self.cur_shard = 0
        shard_name = self.shards[self.cur_shard]
        blob = self.bucket.blob(shard_name)
        data = blob.download_as_bytes()

        if shard_name.endswith(".bin"):
            # Parse modded-nanogpt .bin format from bytes
            header = np.frombuffer(data[: BIN_HEADER_SIZE * 4], dtype=np.int32)
            assert header[0] == BIN_MAGIC_NUMBER, f"Invalid magic number: {header[0]}"
            num_tokens = int(header[2])
            tokens = np.frombuffer(
                data[BIN_HEADER_SIZE * 4 : BIN_HEADER_SIZE * 4 + num_tokens * 2],
                dtype=np.uint16,
            )
        else:
            tokens = np.load(BytesIO(data))
            if not isinstance(tokens, np.ndarray):
                tokens = tokens["arr_0"]

        self.shard_size = len(tokens)
        return tokens


class BlendedCloudDataLoader:
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
        for i, (bucket_name, bucket_prefix, batch_size) in enumerate(
            zip(bucket_names, bucket_prefixes, batch_sizes)
        ):
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

        assert X.shape == (self.D, self.B, self.T)

        return X, Y

    def _calc_batch_sizes(self, B, ratios):
        ratios = np.array(ratios)
        assert B >= len(ratios)
        batch_proportions = jnp.astype(B * ratios / sum(ratios), jnp.int32)
        if sum(batch_proportions) < B:
            diff = B - sum(batch_proportions)
            while diff > 0:
                min_idx = jnp.argmin(batch_proportions)
                batch_proportions = batch_proportions.at[min_idx].add(1)
                diff -= 1
        assert sum(batch_proportions) == B
        return batch_proportions


def load_bin_shard(filepath: str | Path) -> np.ndarray:
    """
    Load a modded-nanogpt .bin shard file.

    Format: 256 int32 header (1024 bytes) followed by uint16 tokens.
    Header[0] = magic number (20240520), Header[2] = num_tokens.
    """
    filepath = Path(filepath)
    with open(filepath, "rb") as f:
        header = np.frombuffer(f.read(BIN_HEADER_SIZE * 4), dtype=np.int32)
        assert header[0] == BIN_MAGIC_NUMBER, f"Invalid magic number: {header[0]}"
        num_tokens = int(header[2])
        tokens = np.frombuffer(f.read(num_tokens * 2), dtype=np.uint16)
    return tokens


class HuggingfaceDataLoader(BaseDataLoader):
    """
    DataLoader for Huggingface hosted datasets
    """

    def __init__(
        self,
        dirpath: str,
        batch_size: int,
        block_size: int,
        device_rank: int,
        label: str | None = None,
        quiet: bool = False,
        start_shard: int = 0,
        start_shard_pos: int = 0,
        hf_repo: str = "kjj0/fineweb100B-gpt2",
        num_train_shards: int | None = None,
        download: bool = True,
    ):
        """
        Args:
            dirpath: Local directory for .bin files
            batch_size: Batch size
            block_size: Sequence length
            device_rank: Number of devices for data parallelism
            label: Filter shards by label ('train' or 'val')
            quiet: Suppress logging
            start_shard: Starting shard index
            start_shard_pos: Starting position within shard
            hf_repo: HuggingFace repo ID for downloading
            num_train_shards: Limit number of train shards (None = all 1030)
            download: If True, download missing shards from HuggingFace
        """
        self.dirpath = Path(dirpath)
        self.dirpath.mkdir(parents=True, exist_ok=True)
        self.hf_repo = hf_repo
        self.num_train_shards = num_train_shards
        self.download = download
        super().__init__(
            batch_size,
            block_size,
            device_rank,
            label,
            quiet,
            start_shard=start_shard,
            start_shard_pos=start_shard_pos,
        )

    def _list_shards(self, label: str | None) -> list[str]:
        """Generate list of shard filenames based on label."""
        shards = []

        if label is None or label == "val":
            shards.append("fineweb_val_000000.bin")

        if label is None or label == "train":
            max_train = self.num_train_shards if self.num_train_shards else 1030
            for i in range(1, max_train + 1):
                shards.append(f"fineweb_train_{i:06d}.bin")

        logger.info(f"HuggingfaceDataloader: {len(shards)} shards ({label or 'all'})")
        return shards

    def _get_shard_path(self, shard_name: str) -> Path:
        """Get local path for shard, downloading from HuggingFace if needed."""
        local_path = self.dirpath / shard_name

        if not local_path.exists() and self.download:
            logger.info(f"Downloading {shard_name} from {self.hf_repo}...")
            try:
                from huggingface_hub import hf_hub_download

                hf_hub_download(
                    repo_id=self.hf_repo,
                    filename=shard_name,
                    repo_type="dataset",
                    local_dir=self.dirpath,
                )
            except Exception as e:
                raise RuntimeError(f"Failed to download {shard_name}: {e}")

        return local_path

    def _load_shard(self) -> np.ndarray:
        """Load current shard, wrapping around if needed."""
        if self.cur_shard >= len(self.shards):
            self.cur_shard = 0

        shard_name = self.shards[self.cur_shard]
        shard_path = self._get_shard_path(shard_name)
        tokens = load_bin_shard(shard_path)
        self.shard_size = len(tokens)
        return tokens
