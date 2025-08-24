import jax
import jax.numpy as jnp
import flax.nnx as nnx

from .rope import calc_rope_omega_llama, apply_rope


class Attention(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config
        self.wq = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02), (None,)
            ),
            use_bias=False,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.wkv = nnx.Linear(
            config.n_embed,
            2 * config.n_kv_head * config.n_embed // config.n_head,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02), (None,)
            ),
            use_bias=False,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.wproj = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.normal(
                    stddev=0.02 * (2 * self.config.n_layer) ** -0.5
                ),
                (None,),
            ),
            use_bias=False,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.rope_omega = nnx.Variable(
            calc_rope_omega_llama(
                config.n_embed // config.n_head,
                config.block_size,
                config.rope_theta,
                config.dtype,
            ),
        )

    def __call__(self, x):
        B, T, C = x.shape
        nH = self.config.n_head
        nKV = self.config.n_kv_head
        q = self.wq(x)
        kv = self.wkv(x)
        k, v = jnp.split(kv, 2, axis=-1)

        q = q.reshape(B, T, nH, C // nH)
        k = k.reshape(B, T, nKV, C // nH)
        v = v.reshape(B, T, nKV, C // nH)

        q = apply_rope(q, self.rope_omega)
        k = apply_rope(k, self.rope_omega)

        implementation = self.config.sdpa_implementation

        match implementation:
            case "cudnn" | "xla":
                y = jax.nn.dot_product_attention(
                    q,
                    k,
                    v,
                    mask=None,
                    bias=None,
                    is_causal=True,
                    implementation=implementation,
                )
            case _:
                _, _, n_head, hs = q.shape
                _, _, n_kv_head, _ = k.shape

                G = n_head // n_kv_head

                q = q.reshape((B, T, n_kv_head, G, hs))  # (B, T, n_kv_head, G, hs)
                q = jnp.transpose(q, axes=(0, 2, 3, 1, 4))

                k = k.reshape(-1, T, n_kv_head, 1, hs)  # (B, T, n_kv_head, 1, hs)
                k = jnp.transpose(k, axes=(0, 2, 3, 4, 1))

                v = v.reshape(-1, T, n_kv_head, 1, hs)  # (B, T, n_kv_head, 1, hs)
                v = jnp.transpose(v, axes=(0, 2, 3, 1, 4))

                att = (q @ k) / jnp.sqrt(hs)  # (B, n_kv_head, G, T, T)

                mask = jnp.tril(jnp.ones((T, T), dtype=jnp.bool))[None, None, None, ...]
                att = jnp.where(mask == False, float("-inf"), att)
                att = jax.nn.softmax(att, axis=-1)
                y = att @ v
                y = y.transpose((0, 3, 1, 2, 4))  # (B, T, n_kv_head, G, hs)
                y = y.reshape(B, T, n_head, hs)  # (B, T, n_head, hs)

        y = jnp.reshape(y, (B, T, C))
        y = self.wproj(y)
        return y


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config

    from rope import calc_rope_omega_llama

    config = Config(n_embed=8, n_head=2)
    B, T = 16, config.n_embed
    rngs = nnx.Rngs(0)
    rope_omega = calc_rope_omega_llama(
        config.n_embed // config.n_head, config.block_size, config.rope_theta
    )
    attn = Attention(config, rope_omega, rngs)
    x = jax.random.normal(jax.random.key(0), (B, T, config.n_embed))
    attn(x)
    assert x.shape == (B, T, config.n_embed)
