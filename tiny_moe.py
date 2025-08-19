# ╔════════════════════════════════════════════════════════════════════════════════════════════════╗
# ║  File    : tiny_moe.py                                                                         ║
# ║  Author  : Vikram Pawar                                                                        ║
# ║  Email   : pvikram035@gmail.com                                                                ║
# ║  Desc    : Minimal Mixture of Experts (MoE) implementation in JAX and Flax NNX.                ║
# ╚════════════════════════════════════════════════════════════════════════════════════════════════╝

from typing import Optional

import jax
import jax.numpy as jnp
import flax.nnx as nnx
import orbax.checkpoint as ocp

from modules.glu import GLU
from modules.attn import Attention
from modules.moe import MoE
from modules.rope import calc_rope_omega_llama

from config import Config


class Tiny_MoE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.aux_loss = False
        self.embedding = nnx.Embed(config.vocab_size, config.n_embed, rngs=rngs)
        self.layers = []
        rope_omega = calc_rope_omega_llama(
            config.n_embed // config.n_head, config.block_size, config.rope_theta
        )
        for _ in range(config.n_layer // 2):
            self.layers.append(Attention(config, rope_omega, rngs))
            self.layers.append(GLU(config, rngs))
            self.layers.append(Attention(config, rope_omega, rngs))
            self.layers.append(MoE(config, rngs))

    def __call__(self, x):
        x = self.embedding(x)
        total_aux_loss = 0
        for i, layer in enumerate(self.layers):
            if self.aux_loss is True and (i+1) % 4 == 0:
                o, aux_loss = layer(x)
                x = x + o
                total_aux_loss += aux_loss
            else:
                x = x + layer(x)
        x = self.embedding.attend(x)
        return x, total_aux_loss

    @staticmethod
    def from_checkpoint(
        fpath: str,
        rngs: nnx.Rngs,
        config: Optional[Config] = None,
        sharding: Optional[jax.sharding.NamedSharding] = None,
    ):

        default = jax.random.key(1337)
        gate_noise = jax.random.key(42)
        rngs = nnx.Rngs(default=default, gate_noise=gate_noise)
        config = config if config else Config()
        abstract_model = nnx.eval_shape(
            lambda: Tiny_MoE(
                config=config, rngs=nnx.Rngs(default=default, gate_noise=gate_noise)
            )
        )
        graphdef, rngstate, other_state = nnx.split(abstract_model, nnx.RngState, ...)
        # pspecs = nnx.get_partition_spec(other_state)
        # sharded_state = nnx.with_sharding_constraint(other_state, pspecs)
        checkpointer = ocp.StandardCheckpointer()
        other_state = checkpointer.restore(fpath, target=other_state)
        model = nnx.merge(graphdef, rngstate, other_state)
        for i in range(len(model.layers)):
            if hasattr(model.layers[i], "moe"):
                # model.h[i].moe.gate_noise_rngstream = rngs["gate_noise"].fork()
                model.layers[i].moe.gate_noise_rngstream = (
                    rngs.gate_noise
                )  # TODO: Temporary fix for backward compatibility with jax 0.5.2
        return model


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
