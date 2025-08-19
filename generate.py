import jax
import jax.numpy as jnp
import flax.nnx as nnx

from transformers import AutoTokenizer

from tiny_moe import Tiny_MoE, Config


@nnx.jit
def _generate_step(m, x, key):
    if type(logits) == tuple:
        logits, _ = m(x)
    else:
        logits = m(x)
    x_new = logits[:, -1, :]
    top_k_vals, top_k_indices = jax.lax.top_k(x_new, 50)
    key, subkey = jax.random.split(key)
    top_k_logit_idxs = jax.random.categorical(subkey, top_k_vals)
    top_k_logit_idxs = top_k_logit_idxs[..., None]  # expand dims
    sample_idxs = jnp.take_along_axis(top_k_indices, top_k_logit_idxs, axis=-1)
    return sample_idxs


def generate(m, x, max_length, key):
    x = jnp.array(x)[None, ...]
    x = jnp.tile(x, (8, 1))
    while x.shape[-1] < max_length:
        sample_idxs = _generate_step(m, x, key)
        x = jnp.concatenate([x, sample_idxs], axis=-1)
    return x


if __name__ == "__main__":
    config = Config()
    m = Tiny_MoE(config, nnx.Rngs(default=0))
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    x = tokenizer.encode("A wise king")
    x = generate(m, x, 10, jax.random.key(1337))
    for i in range(x.shape[0]):
        print(tokenizer.decode(x[i]))
