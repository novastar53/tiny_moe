import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax

print(jax.devices())

mesh = jax.sharding.Mesh(jax.devices(), ["devices"])

import flax.nnx as nnx
import optax

from transformers import AutoTokenizer

from tiny_moe import Tiny_MoE, Config
from generate import generate
from dataloader import Dataloader


def loss_fn(model, x, y):
    logits, aux_loss = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y) + 0.01 * aux_loss
    return loss.mean()


@nnx.jit
def step_fn(model: nnx.Module, optimizer: nnx.Optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss


def train():
    config = Config(
        sdpa_implementation="xla"
    )
    sharding = jax.sharding.NamedSharding(mesh, config.expert_partition_spec)
    with mesh:
        m = Tiny_MoE(config, nnx.Rngs(default=0))
        m.train(add_noise=True, aux_loss=False)
        state = nnx.state(m)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = nnx.with_sharding_constraint(state, pspecs)
        nnx.update(m, sharded_state)

        tx = optax.adam(learning_rate=0.001)
        optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)

        try:
            for e in range(10):
                print(f"epoch", e)
                it = Dataloader(batch_size=16, block_size=config.block_size)()
                for x, y in it:
                    loss = step_fn(model=m, optimizer=optimizer, x=x, y=y)
                print(loss)
        except KeyboardInterrupt:
            print("Done.")
        finally:
            tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            x = tokenizer.encode("A wise king")
            x = generate(m, x, 21, key=jax.random.key(1337))
            for i in range(x.shape[0]):
                print(tokenizer.decode(x[i]))


if __name__ == "__main__":
    train()
