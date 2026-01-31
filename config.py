from typing import Literal

from dataclasses import dataclass

import jax.numpy as jnp
from jax.sharding import PartitionSpec


@dataclass(eq=True, unsafe_hash=True)
class Config:
    name: str = "Tiny_MoE"
    dtype: jnp.dtype = jnp.bfloat16
    vocab_size: int = 50304
    block_size: int = 128
    n_layer: int = 4
    n_embed: int = 576
    n_glu_hidden: int = 1536
    n_head: int = 9
    n_kv_head: int = 3
    n_experts: int = 8
    init_stddev: float = 0.02
    expert_load_factor: float = 2.5
    aux_loss_coeff: float = 0.01
    moe_bias: bool = False
    mlp_bias: bool = False
    attention_bias: bool = False
    load_balance_loss_coeff: float = 1e-2
    z_loss_coeff: float = 5e-4
    expert_top_k: int = 2
    ln_epsilon: float = 1e-5
    rope_theta: float = 1e-4  # base frequency for rope
    expert_partition_spec: PartitionSpec = PartitionSpec(
        "devices",
    )
    sdpa_implementation: Literal["xla", "cudnn", "slow"] = (
        "xla"  # self-attention kernel implementation
    )
    value_residual_init: float = 0.5  # Initial lambda for value residual
