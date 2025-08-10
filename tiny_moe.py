# ╔════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  File    : tiny_moe.py                                                                         ║
# ║  Author  : Vikram Pawar                                                                        ║
# ║  Email   : pvikram035@gmail.com                                                                ║
# ║  Desc    : Minimal Mixture of Experts (MoE) implementation in JAX and Flax NNX.                ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════╝

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax.nnx as nnx

from modules.glu import GLU

@dataclass
class Config:
    vocab_size: int = 49152
    block_size: int = 128
    n_layer: int = 4
    n_embed: int = 8
    n_hidden: int = 16


class Tiny_MoE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.embedding = nnx.Embed(config.vocab_size, config.n_embed, rngs=rngs)
        self.layers = [
            GLU(config, rngs)
            for _ in range(config.n_layer)
        ]
    

    def __call__(self, x):
        x = self.embedding(x)
        for i in range(self.config.n_layer):
            x = self.layers[i](x)
        x = self.embedding.attend(x)
        return x



if __name__ == "__main__":
    config = Config()
    B = 16
    x = jax.random.randint(
        jax.random.key(0), (B, config.block_size), 0, config.vocab_size
    )
    rngs = nnx.Rngs(default=1)
    m = Tiny_MoE(config, rngs)
    y = m(x)
    assert(y.shape == (B, config.block_size, config.vocab_size))