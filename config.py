from dataclasses import dataclass

from jax.sharding import PartitionSpec


@dataclass
class Config:
    vocab_size: int = 49152
    block_size: int = 12
    n_layer: int = 8
    n_embed: int = 64
    n_hidden: int = 64 * 3
    n_head: int = 8
    n_experts: int = 8
    expert_load_factor: int = 2
    expert_top_k: int = 2
    expert_partition_spec: PartitionSpec = PartitionSpec(
        "devices",
    )
