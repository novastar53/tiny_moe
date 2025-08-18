import os

from google.cloud import storage

import jax
import flax.nnx as nnx


def load_checkpoint(model, output_dir, config, run_dirname, step, rngs):
    checkpoint_path = (
        output_dir
        / config.name
        / "checkpoints"
        / run_dirname
        / f"checkpoint-{step}.pt"
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
    total_params = jax.tree_util.tree_reduce(
        lambda x, y: x + y, param_counts, 0
    )

    return total_params

