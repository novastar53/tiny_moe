import jax
from flax.nnx.nn import dtypes
import jax.numpy as jnp
import flax.nnx as nnx


class MoE(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config
        self.capacity = config.block_size // config.n_experts

        ## Router Gate
        w_router_gate_init = nnx.with_partitioning(
                nnx.initializers.normal(stddev=0.02), (None,)
            )
        b_router_gate_init = nnx.with_partitioning(
                nnx.initializers.zeros, (None,)
            )
        
        self.w_router_gate = nnx.Param(
            w_router_gate_init(
                rngs.default(),
                (config.n_experts, self.capacity, config.n_embed),
            )
        )

        if config.mlp_bias:

            self.b_router_gate = nnx.Param(
                b_router_gate_init(
                    rngs.default(),
                    (config.n_experts, self.capacity),
                )
            )

        ## Experts
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
                rngs.default(),
                (config.n_experts, config.n_embed, config.n_glu_hidden),
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

        if config.moe_bias:

            b_init = nnx.with_partitioning(
                nnx.initializers.zeros, sharding=self.config.expert_partition_spec
            )

            self.b_proj = nnx.Param(
                b_init(
                    rngs.default(),
                    (config.n_experts, 1, config.n_embed),
                )
            )

            self.b_gate = nnx.Param(
                b_init(
                    rngs.default(),
                    (config.n_experts, 1, config.n_glu_hidden),
                )
            )

            self.b_fc = nnx.Param(
                b_init(
                    rngs.default(),
                    (config.n_experts, 1, config.n_glu_hidden),
                )
            )

    def _apply_experts(self, x):
        (x, w_fc, w_gate, w_proj) = dtypes.promote_dtype(
            (
                x,
                self.w_fc.value,
                self.w_gate.value,
                self.w_proj.value,
            ),
            dtype=self.config.dtype,
        )
        if self.config.moe_bias:
            (b_fc, b_gate, b_proj) = dtypes.promote_dtype(
                (
                    self.b_fc,
                    self.b_gate,
                    self.b_proj,
                ),
                dtype=self.config.dtype,
            )
        x = jax.lax.with_sharding_constraint(x, self.config.expert_partition_spec)
        g = jnp.einsum("enc,ech->enh", x, w_gate)
        if self.config.moe_bias:
            g += b_gate
        h = jnp.einsum("enc,ech->enh", x, w_fc)
        if self.config.moe_bias:
            h += b_fc
        h = nnx.silu(g) * h
        o = jnp.einsum("enh,ehc->enc", h, w_proj)
        if self.config.moe_bias:
            h += b_proj
        o = jax.lax.with_sharding_constraint(o, self.config.expert_partition_spec)
        return o

    def _dispatch(self, x, w):
        x = jnp.einsum('td,tec->ecd', x, w)
        return x
    
    def _collect(self, x, w):
        x = jnp.einsum('ecd,tec->td', x, w)
        return x

    def __call__(self, x):
        B, T, C = x.shape
        g = jnp.einsum('btd,ecd->btec', x, self.w_router_gate) # B, T, E, capacity
        if self.config.mlp_bias:
            g += self.b_router_gate
        dispatch_weights = jax.nn.softmax(g, axis=1)
        collect_weights = jax.nn.softmax(g, axis=(2,3))
        
        expert_inputs = jax.vmap(lambda x, w: self._dispatch(x, w))(x, dispatch_weights)
        expert_inputs = jax.lax.with_sharding_constraint(
            expert_inputs, self.config.expert_partition_spec
        )  # B, n_experts, expert_cap, C

        if B % self.config.n_experts == 0:
            expert_inputs = expert_inputs.reshape(
                self.config.n_experts,
                -1,
                self.config.n_experts,
                self.capacity,
                C,
            )  # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_inputs = jnp.swapaxes(
                expert_inputs, 0, 2
            )  # n_experts, batch_per_expert, n_experts, expert_cap, C
        else:
            expert_inputs = jnp.swapaxes(
                expert_inputs, 0, 1
            )  # n_experts, B, expert_cap, C

        expert_inputs = expert_inputs.reshape(-1, C)  # n_experts * B * expert_cap, C
        expert_inputs = jax.lax.with_sharding_constraint(
            expert_inputs, self.config.expert_partition_spec
        )
        expert_inputs = expert_inputs.reshape(
            self.config.n_experts, B * self.capacity, C
        )  # n_experts, B * expert_cap, C

        expert_outputs = self._apply_experts(expert_inputs)

        if B % self.config.n_experts == 0:
            expert_outputs = expert_outputs.reshape(
                self.config.n_experts,
                -1,
                self.config.n_experts,
                self.capacity,
                C,
            )  # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_outputs = jnp.swapaxes(
                expert_outputs, 0, 2
            )  # n_experts, batch_per_expert, n_experts, expert_cap, C
            expert_outputs = expert_outputs.reshape(
                B, self.config.n_experts, self.capacity, C
            )  # B, n_experts, expert_cap, C
        else:
            expert_outputs = expert_outputs.reshape(
                self.config.n_experts, B, self.capacity, C
            )  # n_experts, B, expert_cap, C
            expert_outputs = jnp.swapaxes(
                expert_outputs, 0, 1
            )  # B, n_experts, expert_cap, C

        expert_outputs = jax.lax.with_sharding_constraint(
            expert_outputs, self.config.expert_partition_spec
        )

        y = jax.vmap(lambda eo, w: self._collect(eo,w))(
            expert_outputs, collect_weights
        )

        y = jax.lax.with_sharding_constraint(y, self.config.expert_partition_spec)
        return y 


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
        state = nnx.state(moe)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = nnx.with_sharding_constraint(state, pspecs)
        nnx.update(moe, sharded_state)
        x = moe(x)
