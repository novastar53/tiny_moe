import jax
import flax.nnx as nnx


class GLU(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config
        self.fc = nnx.Linear(
            config.n_embed,
            config.n_hidden,
            rngs=rngs
        )
        self.proj = nnx.Linear(
            config.n_hidden,
            config.n_embed,
            rngs=rngs
        )

    
    def __call__(self, x):
        return  self.proj(
            nnx.gelu(
                self.fc(x), approximate=True
            )
        )
        
        