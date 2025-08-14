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
            b_init(rngs.default(), (config.n_experts, config.n_hidden))
        )
        self.w_gate = nnx.Param(
            w_fc_init(
                rngs.default(), (config.n_experts, config.n_embed, config.n_hidden)
            )
        )
        self.b_gate = nnx.Param(
            b_init(rngs.default(), (config.n_experts, config.n_hidden))
        )
        self.w_proj = nnx.Param(
            w_proj_init(
                rngs.default(), (config.n_experts, config.n_hidden, config.n_embed)
            )
        )
        self.b_proj = nnx.Param(
            b_init(rngs.default(), (config.n_experts, config.n_embed))
        )
        self.router_gate = nnx.Linear(config.n_embed, config.n_experts, rngs=rngs)

    def assign_per_batch_experts(self, gate_probs, expert_cap):
        top_k_probs, expert_indices = jax.lax.top_k(
            gate_probs, self.config.expert_top_k
        )  # T, K

    def gather(self, x):
        pass

    def __call__(self, x):
        B, T, C = x.shape
        g = self.router_gate(x)
        gate_probs = jax.nn.softmax(g)
        expert_cap_per_batch = int(
            self.config.expert_load_factor
            * self.config.expert_top_k
            * max(1, T // self.config.n_experts)
        )
        jax.vmap(lambda p: self.assign_per_batch_experts(p, expert_cap_per_batch))(
            gate_probs
        )


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
