import jax
import flax.nnx as nnx


class GLU(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config
        self.fc = nnx.Linear(config.n_embed, config.n_glu_hidden, 
                             kernel_init=nnx.initializers.normal(stddev=0.02), 
                             use_bias=False,
                             dtype=config.dtype,
                             rngs=rngs)
        self.gate = nnx.Linear(config.n_embed, config.n_glu_hidden, 
                               kernel_init=nnx.initializers.normal(stddev=0.02),
                               use_bias=False,
                               dtype=config.dtype,
                               rngs=rngs)
        self.proj = nnx.Linear(config.n_glu_hidden, config.n_embed, 
                               kernel_init=nnx.initializers.normal(stddev=0.02 * (2 * config.n_layer) ** -0.5),
                               use_bias=False,
                               dtype=config.dtype,
                               rngs=rngs)

    def __call__(self, x):
        g = nnx.silu(self.gate(x))
        h = self.fc(x)
        h = g * h
        o = self.proj(h)

        return {
            "output": o
        }
