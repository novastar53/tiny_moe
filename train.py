import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

import jax

print(jax.devices())

jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32") # Set the default precision for matrix multiplication


mesh = jax.sharding.Mesh(jax.devices(), ["devices"])

import flax.nnx as nnx
import optax

import tiktoken
from transformers import AutoTokenizer

from tiny_moe import Tiny_MoE, Config
from generate import generate
from dataloader import Dataloader
from utils import count_params


def loss_fn(model, x, y):
    output = model(x)
    logits = output["output"]
    aux_loss = output["aux_loss"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return loss.mean() + 0.01 * aux_loss


@nnx.jit
def step_fn(model: nnx.Module, optimizer: nnx.Optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss


def train():
    config = Config(
        sdpa_implementation="slow"
    )
    sharding = jax.sharding.NamedSharding(mesh, config.expert_partition_spec)
    with mesh:
        m = Tiny_MoE(config, nnx.Rngs(default=1337, gate_noise=42))
        num_params = count_params(m)
        print(f"Number of parameters: {num_params:,}")
        m.train(add_noise=True, aux_loss=True)
        state = nnx.state(m)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = nnx.with_sharding_constraint(state, pspecs)
        nnx.update(m, sharded_state)

        tx = optax.adam(learning_rate=0.001)
        optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)

        try:
            for e in range(100):
                print(f"epoch", e)
                it = Dataloader(batch_size=16, block_size=config.block_size)()
                for x, y in it:
                    loss = step_fn(model=m, optimizer=optimizer, x=x, y=y)
                print(loss)
        except KeyboardInterrupt:
            print("Done.")
        finally:
            #tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
            tokenizer = tiktoken.get_encoding("gpt2")
            x = tokenizer.encode("The forest lion and the bull")
            m.eval(add_noise=False, aux_loss=False)
            x = generate(m, x, 21, 0.1, jax.random.key(1337))
            for i in range(x.shape[0]):
                print(tokenizer.decode(x[i]))


if __name__ == "__main__":
    train()
