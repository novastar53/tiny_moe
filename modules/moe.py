import jax
import jax.numpy as jnp
import flax.nnx as nnx


class MoE(nnx.Module):
    def __init__(self, config, rngs: nnx.Rngs):
        self.config = config
        self.w_fc = nnx.Param(
            config.n_experts,
            config.n_embed,
            config.n_hidden,
            rngs
        )
        self.b_fc = nnx.Param(
            config.n_experts,
            config.n_hidden,
            rngs
        )
        self.w_gate = nnx.Param(
            config.n_experts,
            config.n_embed,
            config.n_hidden,
            rngs
        )
        self.b_gate = nnx.Param(
            config.n_experts,
            config.n_hidden,
            rngs
        )
        self.w_proj = nnx.Param(
            config.n_experts,
            config.n_hidden,
            config.n_embed,
            rngs
        )
        self.b_proj = nnx.Param(
            config.n_experts,
            config.n_embed,
            rngs
        )
        self.router_gate = nnx.Linear(
           config.n_embed,
           config.n_experts,
           rngs
        )


    def assign_per_batch_experts(self, gate_probs, expert_cap):
        top_k_probs, expert_indices = jax.lax.top_k(gate_probs, self.config.expert_top_k)  # T, K



    def gather(self, x):
        pass


    def __call__(self, x):
        B, T, C = x.shape
        g = self.router_gate(x)
        gate_probs = jax.nn.softmax(g)
        expert_cap_per_batch = int(self.config.expert_load_factor * self.config.expert_top_k * max(1, T // self.config.n_experts))
        jax.vmap(
            lambda p: self.assign_per_batch_experts(p, expert_cap_per_batch)
        )(gate_probs)


if __name__ == "__main__":
    import sys
    import os
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
    from config import Config
    config = Config()
    B, T = 16, 128
    rngs = nnx.Rngs(0)

    moe = MoE(config, rngs)