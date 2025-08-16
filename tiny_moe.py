# ╔════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  File    : tiny_moe.py                                                                         ║
# ║  Author  : Vikram Pawar                                                                        ║
# ║  Email   : pvikram035@gmail.com                                                                ║
# ║  Desc    : Minimal Mixture of Experts (MoE) implementation in JAX and Flax NNX.                ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════╝

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from modules.glu import GLU
from modules.attn import Attention
from modules.moe import MoE

from config import Config


class Tiny_MoE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.embedding = nnx.Embed(config.vocab_size, config.n_embed, rngs=rngs)
        self.layers = []
        for _ in range(config.n_layer // 2):
            self.layers.append(Attention(config, rngs))
            self.layers.append(GLU(config, rngs))
            self.layers.append(Attention(config, rngs))
            self.layers.append(MoE(config, rngs))

    def __call__(self, x):
        x = self.embedding(x)
        for i in range(self.config.n_layer):
            x = x + self.layers[i](x)
        x = self.embedding.attend(x)
        return x


if __name__ == "__main__":
    import os
    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

    print(jax.devices())

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
