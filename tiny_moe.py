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

from modules.attn import Attention
from modules.moe import MoE
from modules.rope import calc_rope_omega_llama

from config import Config


class Block(nnx.Module):
    def __init__(self, config: Config, layer_idx: int, rngs: nnx.Rngs):
        self.config = config
        self.layer_idx = layer_idx
        self.rms_n_1 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, (None,)),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.rms_n_2 = nnx.RMSNorm(
            config.n_embed,
            epsilon=config.ln_epsilon,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, (None,)),
            dtype=config.dtype,
            rngs=rngs,
        )
        self.load_balance_loss = False
        self.z_loss = False
        self.attn = Attention(config, rngs)
        self.moe = MoE(config, rngs)

    def __call__(self, x, v1=None, value_lambda=None):
        attn_out, v1 = self.attn(
            self.rms_n_1(x),
            v1=v1,
            value_lambda=value_lambda,
            layer_idx=self.layer_idx,
        )
        x = x + attn_out
        o = self.moe(self.rms_n_2(x))
        x = x + o["y"]
        o["y"] = x
        return o, v1


class Tiny_MoE(nnx.Module):
    def __init__(self, config: Config, rngs: nnx.Rngs):
        self.config = config
        self.load_balance_loss = False
        self.z_loss = False
        self.embedding = nnx.Embed(
            config.vocab_size,
            config.n_embed,
            embedding_init=nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02), (None,)
            ),
            dtype=config.dtype,
            rngs=rngs,
        )

        self.value_residual_lambdas = nnx.Param(
            jnp.full(config.n_layer, config.value_residual_init, dtype=config.dtype)
        )

        # Skip connection gating parameters
        self.skip_lambda = nnx.Param(
            jnp.array(config.skip_lambda_init, dtype=config.dtype)
        )
        self.skip_gate = nnx.Linear(
            config.skip_gate_input_dim,
            1,
            use_bias=False,
            kernel_init=nnx.initializers.zeros,  # Start with zero gate
            dtype=config.dtype,
            rngs=rngs,
        )

        self.h = [Block(config, layer_idx=i, rngs=rngs) for i in range(config.n_layer)]
        self.rms_n_f = nnx.RMSNorm(
            config.n_embed,
            dtype=config.dtype,
            scale_init=nnx.with_partitioning(nnx.initializers.ones, (None,)),
            rngs=rngs,
        )

    def __call__(self, x):
        x_embed = self.embedding(x)
        x0 = x_embed  # Save original embedding for skip gating
        v1 = None  # Will be captured from first block

        # U-Net skip connection stack (LIFO)
        # Stores full activations: shape (B, T, n_embed)
        skip_connections = []

        total_load_balance_loss = 0
        total_z_loss = 0

        for i in range(self.config.n_layer):
            # Apply skip connection BEFORE block (if this is a skip_out layer)
            if i in self.config.unet_skip_out_layers and skip_connections:
                skip_value = skip_connections.pop()  # (B, T, n_embed)
                # Gated skip connection
                gate = (
                    jax.nn.sigmoid(self.skip_lambda) * 2 *
                    jax.nn.sigmoid(self.skip_gate(x0[..., :self.config.skip_gate_input_dim]))
                )
                x_embed = x_embed + gate * skip_value

            # Process through block
            value_lambda = self.value_residual_lambdas[i]
            out, v1 = self.h[i](x_embed, v1=v1, value_lambda=value_lambda)
            x_embed = out["y"]

            # Save for skip connection AFTER block (if this is a skip_in layer)
            if i in self.config.unet_skip_in_layers:
                skip_connections.append(x_embed)

            if self.load_balance_loss:
                total_load_balance_loss += out["load_balance_loss"]
            if self.z_loss:
                total_z_loss += out["z_loss"]

        x_embed = self.rms_n_f(x_embed)
        logits = self.embedding.attend(x_embed)
        return logits, total_load_balance_loss, total_z_loss


if __name__ == "__main__":
    import os

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

    config = Config(
        unet_skip_in_layers=(0, 1),
        unet_skip_out_layers=(2, 3),
    )
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
        y = m(x)[0]
        assert y.shape == (B, config.block_size, config.vocab_size)
        assert 0 == jnp.count_nonzero(jnp.isnan(y))