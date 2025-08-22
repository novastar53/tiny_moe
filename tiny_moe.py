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


class GLU_Block(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.rms_n_1 = nnx.RMSNorm(
            config.n_embed,
            scale_init=nnx.initializers.ones,
            dtype=config.dtype,
            rngs=rngs
        )
        self.rms_n_2 = nnx.RMSNorm(
            config.n_embed,
            scale_init=nnx.initializers.ones,
            dtype=config.dtype,
            rngs=rngs
        )
        rope_omega = calc_rope_omega_llama(
            config.n_embed // config.n_head, config.block_size, config.rope_theta, config.dtype
        )
        self.attn = Attention(config, rope_omega, rngs)
        self.glu = GLU(config, rngs)
    

    def __call__(self, x):
        o = self.attn(self.rms_n_1(x))
        x = x + o["output"]
        o = self.glu(self.rms_n_2(x))
        x = x + o["output"]
        return {
            "output": x
        }


class MOE_Block(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.rms_n_1 = nnx.RMSNorm(
            config.n_embed,
            scale_init=nnx.initializers.ones,
            dtype=config.dtype,
            rngs=rngs
        )
        self.rms_n_2 = nnx.RMSNorm(
            config.n_embed,
            scale_init=nnx.initializers.ones,
            dtype=config.dtype,
            rngs=rngs
        )
        rope_omega = calc_rope_omega_llama(
            config.n_embed // config.n_head, config.block_size, config.rope_theta, config.dtype
        )
        self.attn = Attention(config, rope_omega, rngs)
        self.moe = MoE(config, rngs)


    def __call__(self, x):
        attn_o = self.attn(self.rms_n_1(x))
        x = x + attn_o["output"]
        moe_o = self.moe(self.rms_n_2(x))
        x = x + moe_o["output"]
        o =  {
            "output": x
        }
        if "aux_loss" in moe_o:
            o["aux_loss"] = moe_o["aux_loss"]
        return o


class Tiny_MoE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.aux_loss = False
        self.embedding = nnx.Embed(config.vocab_size, config.n_embed,
                                   embedding_init=nnx.initializers.normal(stddev=0.02), 
                                   dtype=config.dtype,
                                   rngs=rngs)
        self.h = []
        for _ in range(config.n_layer // 2):
            self.h += [
                MOE_Block(config, rngs=rngs),
                GLU_Block(config, rngs=rngs),
            ]

        self.rms_n_f = nnx.RMSNorm(config.n_embed,
                                   dtype=config.dtype,
                                   scale_init=nnx.initializers.ones, rngs=rngs)
 

    def __call__(self, x):
        x = self.embedding(x)
        total_aux_loss = 0
        for i in range(self.config.n_layer):
            o = self.h[i](x)
            x = o['output'] 
            if "aux_loss" in o:
                total_aux_loss += o["aux_loss"]
        x = self.rms_n_f(x)
        logits = self.embedding.attend(x)
        return {
            "output": logits, 
            "aux_loss": total_aux_loss
        } 


if __name__ == "__main__":
    import os

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

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
        y = m(x)["output"]
        assert y.shape == (B, config.block_size, config.vocab_size)
