import os
import random
from english_words import get_english_words_set
imp

from google.cloud import storage

import jax
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp

from config import Config
from tiny_moe import Tiny_MoE


def generate_readable_code():
    words = [w.lower() for w in get_english_words_set(['web2']) if 4 <= len(w) <= 8]
    return f"{random.choice(words)}_{random.choice(words)}"


@nnx.jit(static_argnums=(0,1))
def create_sharded_model(config: Config, mesh: jax.sharding.Mesh, rngs: nnx.Rngs):
    with mesh:
        m = Tiny_MoE(config, rngs)
        state = nnx.state(m)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = nnx.with_sharding_constraint(state, pspecs)
        nnx.update(m, sharded_state)
        return m


def load_checkpoint(model, output_dir, config, run_dirname, step, rngs):
    checkpoint_path = (
        output_dir / config.name / "checkpoints" / run_dirname / f"checkpoint-{step}.pt"
    )
    m = model.from_checkpoint(checkpoint_path, rngs, config)
    return m


def load_checkpoint_from_gcloud(
    model, config, output_dir, bucket_name, run_dirname, step, rngs
):
    try:
        return load_checkpoint(model, output_dir, config, run_dirname, step, rngs)
    except:
        client = storage.Client()
        prefix = f"{config.name}/checkpoints/{run_dirname}/checkpoint-{step}.pt/"
        checkpoint_path = (
            output_dir
            / config.name
            / "checkpoints"
            / run_dirname
            / f"checkpoint-{step}.pt/"
        )
        for blob in client.list_blobs(bucket_name, prefix=prefix):
            if blob.name.endswith("/"):
                continue
            rel_path = blob.name[len(prefix) :]
            dst_path = os.path.join(checkpoint_path, rel_path)
            os.makedirs(os.path.dirname(dst_path), exist_ok=True)
            blob.download_to_filename(dst_path)
        m = model.from_checkpoint(checkpoint_path, rngs, config)
        return m


def save_checkpoint(m, output_dir, run_dirname, step):
    checkpoint_dirpath = (
        output_dir / m.config.name / "checkpoints" / run_dirname
    )
    checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dirpath / f"checkpoint-{step}.pt"
    print(f"Saving model checkpoint to {checkpoint_path}")
    m.save_checkpoint(checkpoint_path)


def save_optimizer_state(m, optimizer):
  state_dirpath = (
    output_dir / m.config.name / "optimizer_checkpoints" / run_dirname
  )
  state_dirpath.mkdir(parents=True, exist_ok=True)
  _, state = nnx.split(optimizer)
  state.model = None
  cp = ocp.StandardCheckpointer()
  print(f"Saving optimizer state to {state_dirpath}/step-{optimizer.step.value.item()}")
  cp.save(state_dirpath / f"step-{optimizer.step.value.item()}", state)
  cp.wait_until_finished()


def load_optimizer_state(model, optimizer, run_dirname, step):
  cp = ocp.StandardCheckpointer()
  graphdef, state = nnx.split(optimizer)
  state_model = state.model
  state.model = None
  state = cp.restore(output_dir / optimizer.model.config.name / "optimizer_checkpoints" / run_dirname / f"step-{step}", target=state)
  state.model = state_model
  optimizer = nnx.merge(graphdef, state)
  return optimizer


def count_params(m: nnx.Module, layer_type: str | None = None) -> int:
    def get_size(y):
        return y.size

    if layer_type is not None:

        def _filter(path, val):
            return issubclass(val.type, nnx.Param) and layer_type in path

        _, params, _ = nnx.split(m, _filter, nnx.Variable)
    else:
        _, params, _ = nnx.split(m, nnx.Param, nnx.Variable)
    param_counts = jax.tree_util.tree_map(get_size, params)
    total_params = jax.tree_util.tree_reduce(lambda x, y: x + y, param_counts, 0)

    return total_params


def loss_fn(model, x, y):
    output = model(x)
    logits = output["output"]
    aux_loss = output["aux_loss"]
    loss = optax.softmax_cross_entropy_with_integer_labels(logits, y)
    return loss.mean() + model.config.aux_loss_coeff * aux_loss, aux_loss


@nnx.jit
def step_fn(model: nnx.Module, optimizer: nnx.Optimizer, x, y):
    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(model, grads)
    return loss

