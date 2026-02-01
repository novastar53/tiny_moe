import os
from typing import Optional
import random
import csv
from english_words import get_english_words_set

from google.cloud import storage

import jax
import jax.numpy as jnp
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
    words = [w.lower() for w in get_english_words_set(["web2"]) if 4 <= len(w) <= 8]
    return f"{random.choice(words)}_{random.choice(words)}"


@nnx.jit(static_argnums=(0, 1))
def create_sharded_model(config: Config, mesh: jax.sharding.Mesh, rngs: nnx.Rngs):
    with mesh:
        m = Tiny_MoE(config, rngs)
        state = nnx.state(m)
        pspecs = nnx.get_partition_spec(state)
        sharded_state = nnx.with_sharding_constraint(state, pspecs)
        nnx.update(m, sharded_state)
        return m


def _from_checkpoint(
    fpath: str,
    rngs: nnx.Rngs,
    config: Optional[Config] = None,
    sharding: Optional[jax.sharding.NamedSharding] = None,
):

    default = jax.random.key(1337)
    gate_noise = jax.random.key(42)
    rngs = nnx.Rngs(default=default, gate_noise=gate_noise)
    config = config if config else Config()
    abstract_model = nnx.eval_shape(
        lambda: Tiny_MoE(
            config=config, rngs=nnx.Rngs(default=default, gate_noise=gate_noise)
        )
    )
    graphdef, rngstate, other_state = nnx.split(abstract_model, nnx.RngState, ...)
    # pspecs = nnx.get_partition_spec(other_state)
    # sharded_state = nnx.with_sharding_constraint(other_state, pspecs)
    checkpointer = ocp.StandardCheckpointer()
    other_state = checkpointer.restore(fpath, target=other_state)
    model = nnx.merge(graphdef, rngstate, other_state)
    for i in range(len(model.h)):
        if hasattr(model.h[i], "moe"):
            # model.h[i].moe.gate_noise_rngstream = rngs["gate_noise"].fork()
            model.h[i].moe.gate_noise_rngstream = (
                rngs.gate_noise
            )  # TODO: Temporary fix for backward compatibility with jax 0.5.2
    return model


def load_checkpoint(output_dir, config, run_dirname, step, rngs):
    checkpoint_path = (
        output_dir
        / config.name
        / "model_checkpoints"
        / run_dirname
        / f"checkpoint-{step}.pt"
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
        prefix = f"{config.name}/model_checkpoints/{run_dirname}/checkpoint-{step}.pt/"
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
    checkpoint_dirpath = output_dir / m.config.name / "model_checkpoints" / run_dirname
    checkpoint_dirpath.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dirpath / f"checkpoint-{step}.pt"
    print(f"Saving model checkpoint to {checkpoint_path}")
    _, _, other_state = nnx.split(m, nnx.RngState, ...)
    ckptr = ocp.StandardCheckpointer()
    ckptr.save(checkpoint_path, other_state)
    ckptr.wait_until_finished()


def save_optimizer_state(m, output_dir, run_dirname, optimizer):
    state_dirpath = output_dir / m.config.name / "optimizer_checkpoints" / run_dirname
    state_dirpath.mkdir(parents=True, exist_ok=True)
    cp = ocp.StandardCheckpointer()
    print(
        f"Saving optimizer state to {state_dirpath}/step-{optimizer.step.value.item()}"
    )
    state = nnx.state(optimizer)
    cp.save(state_dirpath / f"step-{optimizer.step.value.item()}", state)
    cp.wait_until_finished()


def load_optimizer_state(model, optimizer, run_dirname, output_dir, step):
    cp = ocp.StandardCheckpointer()
    graphdef, state = nnx.split(optimizer)
    state = cp.restore(
        output_dir
        / optimizer.model.config.name
        / "optimizer_checkpoints"
        / run_dirname
        / f"step-{step}",
        target=state,
    )
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
    logits, load_balance_loss, z_loss = model(x)
    logits_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss = (
        logits_loss
        + model.config.aux_loss_coeff * load_balance_loss
        + model.config.z_loss_coeff * z_loss
    )
    return loss, (logits_loss, load_balance_loss, z_loss)


@nnx.jit
def step_fn(model: nnx.Module, optimizer: nnx.Optimizer, x, y):
    (loss, (logits_loss, load_balance_loss, z_loss)), grads = nnx.value_and_grad(
        loss_fn, has_aux=True
    )(model, x, y)
    optimizer.update(model, grads)
    return loss, logits_loss, load_balance_loss, z_loss


@nnx.jit
def compute_val_loss(model, x, y):
    """Compute loss without computing gradients."""
    logits, load_balance_loss, z_loss = model(x)
    logits_loss = optax.softmax_cross_entropy_with_integer_labels(logits, y).mean()
    loss = (
        logits_loss
        + model.config.aux_loss_coeff * load_balance_loss
        + model.config.z_loss_coeff * z_loss
    )
    return loss, logits_loss


def run_validation(model, val_dataloader, data_sharding, num_batches: int = 50):
    """
    Run validation on the model.

    Args:
        model: The model to validate
        val_dataloader: Validation dataloader
        data_sharding: JAX sharding for data
        num_batches: Number of batches to validate on

    Returns:
        Tuple of (avg_loss, avg_logits_loss)
    """
    # Disable noise and aux losses for validation
    model.train(add_noise=False, load_balance_loss=False, z_loss=False)
    total_loss = 0.0
    total_logits_loss = 0.0

    for _ in range(num_batches):
        batch, target = val_dataloader()
        batch = jax.device_put(batch.squeeze(), data_sharding)
        target = jax.device_put(target.squeeze(), data_sharding)
        loss, logits_loss = compute_val_loss(model, batch, target)
        total_loss += loss.item()
        total_logits_loss += logits_loss.item()

    # Re-enable training mode
    model.train(add_noise=False, load_balance_loss=True, z_loss=True)
    return total_loss / num_batches, total_logits_loss / num_batches


def inverse_sqrt_schedule(step, max_lr, warmup_steps):
    """
    Inverse square root learning rate schedule.

    Args:
        step: Current training step
        max_lr: Maximum learning rate
        warmup_steps: Number of warmup steps

    Returns:
        Learning rate for the given step
    """
    warmup_lr = max_lr * (step + 1) / warmup_steps
    regular_lr = max_lr * jnp.sqrt(warmup_steps) / jnp.sqrt(step + 1)
    return jnp.where(step < warmup_steps, warmup_lr, regular_lr)


def plot_lr_schedule(max_steps, max_lr, warmup_ratio, width=80, height=20):
    """
    Plot the inverse square root learning rate schedule using ASCII art.

    Args:
        max_steps: Total number of training steps
        max_lr: Maximum learning rate
        warmup_ratio: Fraction of training for warmup (default 0.01)
        width: Plot width in characters
        height: Plot height in characters
    """
    try:
        import plotext as plt
    except ImportError:
        print("plotext not found. Install with: uv add plotext")
        raise

    warmup_steps = int(max_steps * warmup_ratio)

    # Generate schedule
    steps = jnp.arange(max_steps)
    lrs = inverse_sqrt_schedule(steps, max_lr, warmup_steps)

    # Convert to Python lists for plotext
    steps_list = steps.tolist()
    lrs_list = lrs.tolist()

    # Create plot
    plt.plot_size(width, height)
    plt.plot(steps_list, lrs_list, color="white")

    # Add vertical line marking warmup end
    lr_min = float(lrs.min())
    lr_max = float(lrs.max())
    plt.plot([warmup_steps, warmup_steps], [lr_min, lr_max], color="white")

    # Completely black with white content
    plt.canvas_color("black")
    plt.axes_color("black")
    plt.ticks_color("white")

    # White grid
    plt.grid(True, "white")

    plt.title(f"Inverse Sqrt LR Schedule (max_lr={max_lr}, warmup={warmup_steps})")
    plt.xlabel(
        f"Training Step      ← Warmup End → Cooling Start (Step {warmup_steps:,})"
    )
    plt.ylabel("Learning Rate")

    plt.show()

    # Print statistics with warmup phase information
    max_lr_actual = float(lrs[:warmup_steps].max()) if warmup_steps > 0 else max_lr
    min_lr_actual = (
        float(lrs[warmup_steps:].min()) if warmup_steps < max_steps else max_lr
    )

    print(f"\n{'='*60}")
    print(f"Learning Rate Schedule Statistics:")
    print(f"{'='*60}")
    print(f"  Total steps:     {max_steps:,}")
    print(f"  Warmup phase:    0 → {warmup_steps:,} steps ({warmup_ratio*100:.1f}%)")
    print(
        f"  Cooling phase:   {warmup_steps:,} → {max_steps:,} steps ({(1-warmup_ratio)*100:.1f}%)"
    )
    print(f"  Max LR:          {max_lr_actual:.6f}")
    print(f"  Min LR:          {min_lr_actual:.6f}")
    print(f"  LR decay:        inverse sqrt (1/√step)")
    print(f"{'='*60}\n")
