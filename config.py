from typing import Literal

from dataclasses import dataclass

import jax.numpy as jnp
from jax.sharding import PartitionSpec


@dataclass
class Config:
    name: str = "Tiny_MoE"
    dtype: jnp.dtype = jnp.float32
    vocab_size: int = 50304
    block_size: int = 128
    n_layer: int = 4
    n_embed: int = 576
    n_hidden: int = 1536
    n_head: int = 9
    n_kv_head: int = 3
    n_experts: int = 8
    expert_load_factor: float = 2.5
    aux_loss_coeff: float = 0.01
    expert_top_k: int = 2
    rope_theta: float = 1e-4  # base frequency for rope
    expert_partition_spec: PartitionSpec = PartitionSpec(
        "devices",
    )
    sdpa_implementation: Literal["xla", "cudnn", "slow"] = (
        "xla"  # self-attention kernel implementation
    )
