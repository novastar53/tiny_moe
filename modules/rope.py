import jax.numpy as jnp
import flax.nnx as nnx


def calc_rope_omega_llama(
    n_dim: int,
    block_size: int,
    rope_base_freq: float,
    dtype: jnp.dtype = jnp.float32,
) -> nnx.Variable:
    pow = jnp.arange(0, n_dim, 2, dtype=dtype)
    omega = rope_base_freq ** (pow / n_dim)
    omega = jnp.concat([omega, omega], axis=0)
    pos = jnp.arange(0, block_size, dtype=dtype)
    pos = jnp.expand_dims(pos, axis=1)
    omega = omega * pos
    return nnx.Variable(omega)


def rotate_half(x):
    n = x.shape[-1] // 2
    return jnp.concat((-x[..., n:], x[..., :n]), axis=-1)


def apply_rope(v, omega, offset=0):
    v = v.swapaxes(1, 2)
    omega = omega[offset : offset + v.shape[-2], :]
    a = v * jnp.cos(omega)
    b = rotate_half(v) * jnp.sin(omega)
    y = a + b
    y = y.swapaxes(1, 2)
    return y


if __name__ == "__main__":
    omega = calc_rope_omega_llama(
        n_dim=6,
        block_size=12,
        rope_base_freq=1e-5,
    )

    print(omega.shape)
    print(omega)
