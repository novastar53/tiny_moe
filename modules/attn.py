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
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                (None,),
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
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                (None,),
            ),
            use_bias=False,
            dtype=config.dtype,
            rngs=rngs,
        )
        self.wproj = nnx.Linear(
            config.n_embed,
            config.n_embed,
            kernel_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                (None,),
            ),
            bias_init=nnx.with_partitioning(
                nnx.initializers.zeros,
                (None,),
            ),
            use_bias=False,
            dtype=config.dtype,
            rngs=rngs,
        )
        head_dim = config.n_embed // config.n_head
        self.q_norm = nnx.RMSNorm(
            head_dim,
            epsilon=config.ln_epsilon,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, (None,)),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.k_norm = nnx.RMSNorm(
            head_dim,
            epsilon=config.ln_epsilon,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, (None,)),
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

    def __call__(self, x, v1=None, value_lambda=None, layer_idx=None):
        B, T, C = x.shape
        nH = self.config.n_head
        nKV = self.config.n_kv_head
        q = self.wq(x)
        kv = self.wkv(x)
        k, v = jnp.split(kv, 2, axis=-1)

        q = q.reshape(B, T, nH, C // nH)
        k = k.reshape(B, T, nKV, C // nH)
        v = v.reshape(B, T, nKV, C // nH)

        if layer_idx == 0 and v1 is None:
            v1 = v  
        elif v1 is not None and value_lambda is not None and layer_idx > 0:
            value_lambda = jnp.asarray(value_lambda, dtype=v.dtype)
            v = v + (1 - value_lambda) * v1

        q = self.q_norm(q)
        k = self.k_norm(k)

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
        return y, v1


if __name__ == "__main__":
    import sys
    import os

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config

    config = Config(n_embed=8, n_head=4, n_kv_head=2)
    B, T = 16, config.block_size
    rngs = nnx.Rngs(0)
    attn = Attention(config, rngs)
    x = jax.random.normal(jax.random.key(0), (B, T, config.n_embed))

    # Test first block (captures v1)
    y, v1_captured = attn(x, layer_idx=0)
    assert v1_captured is not None
    assert y.shape == (B, T, config.n_embed)
    print("✓ First block captures v1")

    # Test subsequent block (uses v1)
    y2, v1_returned = attn(x, v1=v1_captured, value_lambda=0.5, layer_idx=1)
    assert v1_returned is v1_captured
    assert y2.shape == (B, T, config.n_embed)
    print("✓ Subsequent block uses v1")

    # Test without v1 (backward compatibility)
    y3, v1_none = attn(x, layer_idx=2)
    assert y3.shape == (B, T, config.n_embed)
    print("✓ Backward compatibility works")

    print("\nAll attention tests passed!")
