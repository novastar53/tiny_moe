import jax
import flax.nnx as nnx
import optax

from transformers import AutoTokenizer

from tiny_moe import Tiny_MoE, Config
from dataloader import Dataloader


def loss_fn(model, state, x, y):
    logits = model(x)
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return loss.mean()


def step_fn(model: nnx.Module, optimizer: nnx.Optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, optimizer, x, y)
    optimizer.update(model, grads)
    return loss


def train():
    config = Config()
    m = Tiny_MoE(config, nnx.Rngs(default=0))
    tx = optax.adam(learning_rate=0.01)
    optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)

    try:
        for e in range(100):
            print(f"epoch", 0)
            it = Dataloader(batch_size=16, block_size=128)()
            for x, y in it:
                loss = step_fn(model=m, optimizer=optimizer, x=x, y=y)
                print(loss)
    except KeyboardInterrupt:
        print("Done.")


if __name__ == "__main__":
    train()