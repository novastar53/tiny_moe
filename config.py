from typing import Literal

from dataclasses import dataclass

import jax.numpy as jnp
from jax.sharding import PartitionSpec


@dataclass
class Config:
    name: str = "Tiny_MoE"
    dtype: jnp.dtype = jnp.float32
    vocab_size: int = 49152
    block_size: int = 128
    n_layer: int = 8
    n_embed: int = 32
    n_hidden: int = n_embed * 3
    n_head: int = 8
    n_kv_head: int = 4
    n_experts: int = 8
    expert_load_factor: int = 2
    expert_top_k: int = 2
    rope_theta: float = 1e-4  # base frequency for rope
    expert_partition_spec: PartitionSpec = PartitionSpec(
        "devices",
    )
    sdpa_implementation: Literal["xla", "cudnn", "slow"] = (
        "xla"  # self-attention kernel implementation
    )
