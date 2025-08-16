import jax
import jax.numpy as jnp
import flax.nnx as nnx


class MoE(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config

        w_fc_init = nnx.with_partitioning(
            nnx.initializers.normal(stddev=0.02),
            sharding=self.config.expert_partition_spec,
        )

        b_init = nnx.with_partitioning(
            nnx.initializers.zeros, sharding=self.config.expert_partition_spec
        )

        w_proj_init = nnx.with_partitioning(
            nnx.initializers.normal(stddev=0.02 * (2 * config.n_layer) ** -0.5),
            sharding=self.config.expert_partition_spec,
        )

        self.w_fc = nnx.Param(
            w_fc_init(
                rngs.default(), (config.n_experts, config.n_embed, config.n_hidden)
            )
        )
        self.b_fc = nnx.Param(
            b_init(rngs.default(), (config.n_experts, 1, config.n_hidden))
        )
        self.w_gate = nnx.Param(
            w_fc_init(
                rngs.default(), (config.n_experts, config.n_embed, config.n_hidden)
            )
        )
        self.b_gate = nnx.Param(
            b_init(rngs.default(), (config.n_experts, 1, config.n_hidden))
        )
        self.w_proj = nnx.Param(
            w_proj_init(
                rngs.default(), (config.n_experts, config.n_hidden, config.n_embed)
            )
        )
        self.b_proj = nnx.Param(
            b_init(rngs.default(), (config.n_experts, 1, config.n_embed))
        )
        self.router_gate = nnx.Linear(config.n_embed, config.n_experts, rngs=rngs)


    def _apply_experts(self, x):
        g = jnp.einsum('enc,ech->enh', x, self.w_gate) + self.b_gate
        h = jnp.einsum('enc,ech->enh', x, self.w_fc) + self.b_fc
        h = nnx.silu(g) * h
        o = jnp.einsum('enh,ehc->enc', h, self.w_proj) + self.b_proj
        o = jax.lax.with_sharding_constraint(o, self.config.expert_partition_spec)
        return o


    def assign_per_batch_experts(self, x, gate_probs, expert_cap):
        _, C = x.shape
        top_k_probs, expert_indices = jax.lax.top_k(
            gate_probs, self.config.expert_top_k
        )  # T, K
        expert_indices = expert_indices.swapaxes(0,1).ravel()
        expert_one_hot = jax.nn.one_hot(expert_indices, self.config.n_experts, dtype=jnp.int32)
        expert_positions = jnp.cumsum(expert_one_hot, axis=0) * expert_one_hot
        expert_positions = expert_positions.reshape(self.config.expert_top_k, -1, self.config.n_experts)
        expert_positions = expert_positions.swapaxes(0,1)
        expert_positions = jnp.max(expert_positions, axis=-1) - 1
        expert_indices = expert_indices.reshape(self.config.expert_top_k, -1).swapaxes(0,1)

        zeros = jnp.zeros((self.config.n_experts, expert_cap, C))
        x = jnp.repeat(x, self.config.expert_top_k, axis=0)
        expert_inputs = zeros.at[expert_indices.ravel(),
                                 expert_positions.ravel()].set(x)
        
        return top_k_probs, expert_positions, expert_indices, expert_inputs
    

    def _collect_outputs(self, expert_outputs, expert_indices, expert_positions, top_k_probs):
        expert_outputs = expert_outputs[expert_indices, expert_positions]
        expert_outputs = jnp.sum(top_k_probs[..., None] * expert_outputs, axis=1)
        return expert_outputs


    def __call__(self, x):
        B, T, C = x.shape
        g = self.router_gate(x)
        gate_probs = jax.nn.softmax(g)
        expert_cap_per_batch = int(
            self.config.expert_load_factor
            * self.config.expert_top_k
            * max(1, T / self.config.n_experts)
        )
        (top_k_probs, 
         expert_positions, 
         expert_indices, 
         expert_inputs) = jax.vmap(
             lambda x, p: self.assign_per_batch_experts(x, p, expert_cap_per_batch)
        )(x, gate_probs) # B, n_experts, expert_cap, C

        expert_inputs = expert_inputs.swapaxes(0, 1) # n_experts, B, expert_cap_per_batch, C
        expert_inputs = jax.lax.with_sharding_constraint(expert_inputs, self.config.expert_partition_spec)
        expert_inputs = expert_inputs.reshape(self.config.n_experts, B * expert_cap_per_batch, C) # n_experts, expert_cap, C
        expert_outputs = self._apply_experts(expert_inputs)
        expert_outputs = expert_outputs.reshape(self.config.n_experts, B, expert_cap_per_batch, C)
        expert_outputs = expert_outputs.swapaxes(0, 1) # B, n_experts, expert_cap_per_batch, C
        y_pred = jax.vmap(
            lambda eo, ei, ep, topk: self._collect_outputs(eo, ei, ep, topk)
        )(expert_outputs, expert_indices, expert_positions, top_k_probs)
        y_pred = jax.lax.with_sharding_constraint(y_pred, self.config.expert_partition_spec)
        
        return y_pred


if __name__ == "__main__":
    import os

    os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
    import sys

    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    from config import Config

    print(jax.devices())

    config = Config()
    B, T = 16, config.block_size
    rngs = nnx.Rngs(0)

    x = jax.random.normal(jax.random.key(1), (B, T, config.n_embed))
    mesh = jax.sharding.Mesh(jax.devices(), ["devices"])
    sharding = jax.sharding.NamedSharding(mesh, config.expert_partition_spec)
    with mesh:
        x = nnx.with_sharding_constraint(x, sharding)
        print(x.device)
        moe = MoE(config, rngs)
        state = nnx.state(moe)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = nnx.with_sharding_constraint(state, pspecs)
        nnx.update(moe, sharded_state)
        print(moe.w_fc.device)
        moe(x)
