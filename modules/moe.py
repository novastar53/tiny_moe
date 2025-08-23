import jax
from flax.nnx.nn import dtypes
import jax.numpy as jnp
import flax.nnx as nnx


class MoE(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config
        self.add_noise = False
        self.aux_loss = False
        self.gate_noise_rngstream = rngs.gate_noise

        self.router_gate = nnx.Linear(config.n_embed, config.n_experts, 
                                      kernel_init=nnx.with_partitioning(
                                          nnx.initializers.normal(stddev=0.02),
                                          (None,)),
                                      use_bias=False,
                                      dtype=config.dtype, 
                                      rngs=rngs)

        w_fc_init = nnx.with_partitioning(
            nnx.initializers.normal(stddev=0.02),
            sharding=self.config.expert_partition_spec,
        )

        w_proj_init = nnx.with_partitioning(
            nnx.initializers.normal(stddev=0.02 * (2 * config.n_layer) ** -0.5),
            sharding=self.config.expert_partition_spec,
        )

        self.w_fc = nnx.Param(
            w_fc_init(
                rngs.default(), (config.n_experts, config.n_embed, config.n_glu_hidden), 
            )
        )
        self.w_gate = nnx.Param(
            w_fc_init(
                rngs.default(), (config.n_experts, config.n_embed, config.n_glu_hidden)
            )
        )
        self.w_proj = nnx.Param(
            w_proj_init(
                rngs.default(), (config.n_experts, config.n_glu_hidden, config.n_embed)
            )
        )

    def _apply_experts(self, x):
        (x, w_fc, w_gate, w_proj) = dtypes.promote_dtype(
        (x, self.w_fc.value, self.w_gate.value, self.w_proj.value), dtype=self.config.dtype)
        x = jax.lax.with_sharding_constraint(x, self.config.expert_partition_spec)
        g = jnp.einsum("enc,ech->enh", x, w_gate)
        h = jnp.einsum("enc,ech->enh", x, w_fc)
        h = nnx.silu(g) * h
        o = jnp.einsum("enh,ehc->enc", h, w_proj)
        o = jax.lax.with_sharding_constraint(o, self.config.expert_partition_spec)
        return o


    def assign_per_batch_experts(self, x, gate_probs, expert_cap):
        _, C = x.shape
        top_k_probs, expert_indices = jax.lax.top_k(
            gate_probs, self.config.expert_top_k
        )  # T, K
        expert_indices = expert_indices.swapaxes(0, 1).ravel()
        expert_one_hot = jax.nn.one_hot(
            expert_indices, self.config.n_experts, dtype=jnp.int32
        )
        expert_positions = jnp.cumsum(expert_one_hot, axis=0) * expert_one_hot
        expert_positions = expert_positions.reshape(
            self.config.expert_top_k, -1, self.config.n_experts
        )
        expert_positions = expert_positions.swapaxes(0, 1)
        expert_positions = jnp.max(expert_positions, axis=-1) - 1
        expert_indices = expert_indices.reshape(self.config.expert_top_k, -1).swapaxes(
            0, 1
        )

        zeros = jnp.zeros((self.config.n_experts, expert_cap, C))
        x = jnp.repeat(x, self.config.expert_top_k, axis=0)
        expert_inputs = zeros.at[expert_indices.ravel(), expert_positions.ravel()].set(
            x
        )

        return top_k_probs, expert_positions, expert_indices, expert_inputs

    def _collect_outputs(
        self, expert_outputs, expert_indices, expert_positions, top_k_probs
    ):
        expert_outputs = expert_outputs[expert_indices, expert_positions]
        expert_outputs = jnp.sum(top_k_probs[..., None] * expert_outputs, axis=1)
        return expert_outputs

    def __call__(self, x):
        B, T, C = x.shape
        g = self.router_gate(x)
        if self.add_noise:
            noise = (
                jax.random.normal(self.gate_noise_rngstream(), g.shape, dtype=self.config.dtype)
                / self.config.n_experts
            )
            g += noise

        gate_probs = jax.nn.softmax(g)
        expert_cap_per_batch = int(
            self.config.expert_load_factor
            * self.config.expert_top_k
            * max(1, T / self.config.n_experts)
        )
        (top_k_probs, expert_positions, expert_indices, expert_inputs) = jax.vmap(
            lambda x, p: self.assign_per_batch_experts(x, p, expert_cap_per_batch)
        )(
            x, gate_probs
        )  # B, n_experts, expert_cap, C

        top_k_probs = jax.lax.with_sharding_constraint(top_k_probs, self.config.expert_partition_spec)
        expert_positions = jax.lax.with_sharding_constraint(expert_positions, self.config.expert_partition_spec)
        expert_indices = jax.lax.with_sharding_constraint(expert_indices, self.config.expert_partition_spec)
        expert_inputs = jax.lax.with_sharding_constraint(expert_inputs, self.config.expert_partition_spec) # B, n_experts, expert_cap, C

        if B % self.config.n_experts == 0:
            expert_inputs = expert_inputs.reshape(self.config.n_experts, -1, self.config.n_experts, expert_cap_per_batch, C) # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_inputs = jnp.swapaxes(expert_inputs, 0, 2) # n_experts, batch_per_expert, n_experts, expert_cap, C
        else:
            expert_inputs = jnp.swapaxes(expert_inputs, 0, 1) # n_experts, B, expert_cap, C

        expert_inputs = expert_inputs.reshape(-1, C) # n_experts * B * expert_cap, C
        expert_inputs = jax.lax.with_sharding_constraint(expert_inputs, self.config.expert_partition_spec)
        expert_inputs = expert_inputs.reshape(self.config.n_experts, B * expert_cap_per_batch, C) # n_experts, B * expert_cap, C

        expert_outputs = self._apply_experts(expert_inputs)

        if B % self.config.n_experts == 0:
            expert_outputs = expert_outputs.reshape(self.config.n_experts, -1, self.config.n_experts, expert_cap_per_batch, C) # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_outputs = jnp.swapaxes(expert_outputs, 0, 2) # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_outputs = expert_outputs.reshape(B, self.config.n_experts, expert_cap_per_batch, C) # B, n_experts, expert_cap, C
        else:
            expert_outputs = expert_outputs.reshape(self.config.n_experts, B, expert_cap_per_batch, C) # n_experts, B, expert_cap, C
            expert_outputs = jnp.swapaxes(expert_outputs, 0, 1) # B, n_experts, expert_cap, C

        expert_outputs = jax.lax.with_sharding_constraint(expert_outputs, self.config.expert_partition_spec)

        y_pred = jax.vmap(
            lambda eo, ei, ep, topk: self._collect_outputs(eo, ei, ep, topk)
        )(expert_outputs, expert_indices, expert_positions, top_k_probs)

        y_pred = jax.lax.with_sharding_constraint(
            y_pred, self.config.expert_partition_spec
        )

        if self.aux_loss is True:
            frac_tokens = jnp.bincount(
                expert_indices.flatten(), length=self.config.n_experts
            ) / (2 * B * T)
            frac_router_probs = jnp.sum(gate_probs, axis=(0, 1)) / (B * T)
            aux_loss = jnp.sum(frac_tokens * frac_router_probs) * self.config.n_experts
            return y_pred, aux_loss

        return y_pred


if __name__ == "__main__":
    import os

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config

    config = Config()
    B, T = 16, config.block_size
    rngs = nnx.Rngs(default=0, gate_noise=1)

    x = jax.random.normal(jax.random.key(1), (B, T, config.n_embed))
    mesh = jax.sharding.Mesh(jax.devices(), ["devices"])
    sharding = jax.sharding.NamedSharding(mesh, config.expert_partition_spec)
    with mesh:
        x = nnx.with_sharding_constraint(x, sharding)
        moe = MoE(config, rngs)
        moe.add_noise = True
        state = nnx.state(moe)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = nnx.with_sharding_constraint(state, pspecs)
        nnx.update(moe, sharded_state)
        x = moe(x)
