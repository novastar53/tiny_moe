import jax
import jax.numpy as jnp
import flax.nnx as nnx


class Attention(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config
        self.qkv = nnx.Linear(config.n_embed, 3 * config.n_embed, rngs=rngs)

    def __call__(self, x):
        B, T, C = x.shape
        nH = self.config.n_head
        qkv = self.qkv(x)  # B, T, 3 * C
        q, k, v = jnp.split(qkv, 3, axis=-1)  # B, T, C

        q = q.reshape(B, T, nH, C // nH)
        k = k.reshape(B, T, nH, C // nH)
        v = v.reshape(B, T, nH, C // nH)

        q = jnp.swapaxes(q, 1, 2)  # B, nH, T, C // nH
        k = jnp.swapaxes(k, 1, 2)  # B, nH, T, C // nH
        k = jnp.swapaxes(k, 2, 3)  # B, nH, C // nH, T
        v = jnp.swapaxes(v, 1, 2)  # B, nH, T, C // nH

        att = (q @ k) / jnp.sqrt(q.shape[-1])  # B, nH, T, T
        mask = jnp.tril(jnp.ones((T, T)))[None, None, ...]
        att = jnp.where(mask == 0.0, float("-inf"), att)
        att = jax.nn.softmax(att, axis=-1)
        x = att @ v  # B, nH, T, C// nH
        x = jnp.swapaxes(x, 1, 2)  # B, T, nH, C // nH
        x = jnp.reshape(x, (B, T, C))
        return x


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config

    config = Config(n_embed=8, n_head=2)
    B, T = 16, 128
    rngs = nnx.Rngs(0)
    attn = Attention(config, rngs)
    x = jax.random.normal(jax.random.key(0), (B, T, config.n_embed))
    attn(x)
    assert x.shape == (B, T, config.n_embed)
