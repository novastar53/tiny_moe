import os
from typing import Optional
import random
import csv
from english_words import get_english_words_set

from google.cloud import storage

import jax
import flax.nnx as nnx
import optax
import orbax.checkpoint as ocp

from config import Config
from tiny_moe import Tiny_MoE


def append_to_csv(file_path, row):
    with open(file_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(row)


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


def _from_checkpoint(
    fpath: str, rngs: nnx.Rngs, 
    config: Optional[Config] = None, 
    sharding: Optional[jax.sharding.NamedSharding] = None
):

    default = jax.random.key(1337)
    gate_noise = jax.random.key(42)
    rngs = nnx.Rngs(default=default, gate_noise=gate_noise)
    config = config if config else Tiny_MoE_Config()
    abstract_model = nnx.eval_shape( 
        lambda: Tiny_MoE(config=config, rngs=nnx.Rngs(default=default, gate_noise=gate_noise))
    )
    graphdef, rngstate, other_state = nnx.split(
        abstract_model, nnx.RngState, ...
    )
    #pspecs = nnx.get_partition_spec(other_state)
    #sharded_state = nnx.with_sharding_constraint(other_state, pspecs)
    checkpointer = ocp.StandardCheckpointer()
    other_state = checkpointer.restore(fpath, target=other_state)
    model = nnx.merge(graphdef, rngstate, other_state)
    for i in range(len(model.h)):
        if hasattr(model.h[i], "moe"):
            #model.h[i].moe.gate_noise_rngstream = rngs["gate_noise"].fork()
            model.h[i].moe.gate_noise_rngstream = rngs.gate_noise # TODO: Temporary fix for backward compatibility with jax 0.5.2
    return model


def load_checkpoint(model, output_dir, config, run_dirname, step, rngs):
    checkpoint_path = (
        output_dir / config.name / "checkpoints" / run_dirname / f"checkpoint-{step}.pt"
    )
    m = _from_checkpoint(checkpoint_path, rngs, config)
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
        m = _from_checkpoint(checkpoint_path, rngs, config)
        return m


def save_checkpoint(m, output_dir, run_dirname, step):
    checkpoint_dirpath = (
        output_dir / m.config.name / "model_checkpoints" / run_dirname
    )
    checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dirpath / f"checkpoint-{step}.pt"
    print(f"Saving model checkpoint to {checkpoint_path}")
    _, _, other_state = nnx.split(m, nnx.RngState, ...)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(checkpoint_path, other_state)
    ckptr.wait_until_finished()


def save_optimizer_state(m, output_dir, run_dirname, optimizer):
  state_dirpath = (
    output_dir / m.config.name / "optimizer_checkpoints" / run_dirname
  )
  state_dirpath.mkdir(parents=True, exist_ok=True)
  cp = ocp.StandardCheckpointer()
  print(f"Saving optimizer state to {state_dirpath}/step-{optimizer.step.value.item()}")
  state = nnx.state(optimizer)
  cp.save(state_dirpath / f"step-{optimizer.step.value.item()}", state)
  cp.wait_until_finished()



def load_optimizer_state(model, optimizer, run_dirname, step):
  cp = ocp.StandardCheckpointer()
  graphdef, state = nnx.split(optimizer)
  state = cp.restore(output_dir / optimizer.model.config.name / "optimizer_checkpoints" / run_dirname / f"step-{step}", target=state)
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
    (loss, aux_loss), grads = nnx.value_and_grad(loss_fn, has_aux=True)(model, x, y)
    optimizer.update(model, grads)
    return loss, aux_loss

