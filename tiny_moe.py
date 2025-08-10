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
from config import Config


class Tiny_MoE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.embedding = nnx.Embed(config.vocab_size, config.n_embed, rngs=rngs)
        self.layers = []
        for _ in range(config.n_layer):
            self.layers.append(Attention(config, rngs))
            self.layers.append(GLU(config, rngs))
    

    def __call__(self, x):
        x = self.embedding(x)
        for i in range(self.config.n_layer):
            x = x + self.layers[i](x)
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