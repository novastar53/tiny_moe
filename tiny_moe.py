# ╔════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  File    : tiny_moe.py                                                                         ║
# ║  Author  : Vikram Pawar                                                                        ║
# ║  Email   : pvikram035@gmail.com                                                                ║
# ║  Desc    : Minimal Mixture of Experts (MoE) implementation in JAX and Flax NNX.                ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════╝

from typing import Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint as ocp

from modules.attn import Attention
from modules.moe import MoE
from modules.rope import calc_rope_omega_llama

from config import Config


class Block(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.rms_n_1 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, (None,)),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.rms_n_2 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, (None,)),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.attn = Attention(config, rngs)
        self.moe = MoE(config, rngs)

    def __call__(self, x):
        x = x + self.attn(self.rms_n_1(x))
        x = x + self.moe(self.rms_n_2(x))
        return x


class Tiny_MoE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.embedding = nnx.Embed(
            config.vocab_size,
            config.n_embed,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02), (None,)
            ),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.h = [Block(config, rngs=rngs) for _ in range(config.n_layer)]
        self.rms_n_f = nnx.RMSNorm(
            config.n_embed,
            dtype=config.dtype,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, (None,)),
            rngs=rngs,
        )

    def __call__(self, x):
        x = self.embedding(x)
        for i in range(self.config.n_layer):
            x = self.h[i](x)
        x = self.rms_n_f(x)
        logits = self.embedding.attend(x)
        return logits


if __name__ == "__main__":
    import os

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

    config = Config()
    mesh = jax.sharding.Mesh(jax.devices(), ["devices"])
    sharding = jax.sharding.NamedSharding(mesh, config.expert_partition_spec)

    B = 16
    x = jax.random.randint(
        jax.random.key(0), (B, config.block_size), 0, config.vocab_size
    )
    rngs = nnx.Rngs(default=1)
    with mesh:
        m = Tiny_MoE(config, rngs)
        state = nnx.state(m)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = nnx.with_sharding_constraint(state, pspecs)
        nnx.update(m, sharded_state)
        y = m(x)
        assert y.shape == (B, config.block_size, config.vocab_size)
