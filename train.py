#!/usr/bin/env python
# coding: utf-8
# Let's Train Tiny MoE
# Author: Vikram Pawar pvikram035 [at] gmail.com

import os

#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"

from datetime import datetime
from pathlib import Path
from pprint import pprint
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import optax

import tiktoken
from transformers import AutoTokenizer

from tiny_moe import Tiny_MoE, Config
from generate import generate
from dataloader import BlendedCloudDataLoader
from utils import (
    generate_readable_code, count_params, create_sharded_model, 
    save_checkpoint, save_optimizer_state, step_fn
)


jax.print_environment_info()
print("Flax version: ", flax.__version__)
print("Optax version: ", optax.__version__)
device = jax.default_backend()
print(f"Platform: {device}")
devices = jax.devices()
num_devices = len(jax.devices())
print(f"Num Devices: {num_devices}")
print("Devices:", jax.devices())


# Set the default precision for matrix multiplication
#####################################
##        jax.lax matmul presets   ##
#####################################
## 'ANY_F8_ANY_F8_F32',
## 'ANY_F8_ANY_F8_F32_FAST_ACCUM'
## 'ANY_F8_ANY_F8_ANY'
## 'ANY_F8_ANY_F8_ANY_FAST_ACCUM'
## 'F16_F16_F16'
## 'F16_F16_F32'
## 'BF16_BF16_BF16'
## 'BF16_BF16_F32'
## 'BF16_BF16_F32_X3'
## 'BF16_BF16_F32_X6'
## 'TF32_TF32_F32'
## 'TF32_TF32_F32_X3'
## 'F32_F32_F32'
## 'F64_F64_F64'
#####################################

# Create model

rngs = nnx.Rngs(default=jax.random.key(1337), 
                gate_noise=jax.random.key(42))

config = Config(
            name="Tiny_MoE",
            dtype=jnp.bfloat16, \
            vocab_size=49152,
            n_layer=4,
            block_size=2048,
            n_head=9,
            n_kv_head=3,
            n_embed=576,
            n_glu_hidden=1536,
            expert_load_factor=1.1,
            sdpa_implementation="cudnn" if device=="gpu" else "xla")
pprint(config)

mesh = jax.sharding.Mesh(jax.devices(), ["devices"])
m = create_sharded_model(config, mesh, rngs)

graphdef, rngstate, state = nnx.split(m, nnx.RngState, ...)
total_params = count_params(m)
moe_params = count_params(m, "moe")

print(f"Parameter Count: {total_params:,}")
print(f"Sharded / MoE Parameter Count: {moe_params:,}")
print(f"Replicated Parameter Count: {total_params - moe_params:,}")

# Set up training config

@dataclass
class TrainerConfig:
  num_tokens: int = int(228e9)
  num_tokens_per_batch: int = 2**15 # 2**19, 0.5 million as per the GPT 3.5 paper
  mB: int = 32 * num_devices
  T: int = 128
  max_steps: int = int(num_tokens // num_tokens_per_batch)
  max_lr: float = 6e-4
  min_lr: float = max_lr * 0.1
  max_grad_norm: float = 1.0  # Clip gradients to this norm
  weight_decay: float = 0.1 # Weight decay for adamw
  adam_b1: float = 0.9
  adam_b2: float = 0.95
  warmup_steps: int = 9000
  print_interval: int = 100
  eval_interval: int = 5000
  checkpoint_interval: int = 10000
  grad_accumulation_steps: int = num_tokens_per_batch // (mB * T) # Number of steps over which to average the gradient

trconf = TrainerConfig()

# Set up optimizer

def trapezoidal_schedule(trconf, step):
    warmup_lr = trconf.max_lr * (step + 1) / trconf.warmup_steps
    cooldown_lr = trconf.max_lr * (trconf.max_steps - step - 1) / (0.2 * trconf.max_steps)

    return jnp.where(step < trconf.warmup_steps,
                     warump_lr,
                     jnp.where(step < 0.8 * trconf.max_steps,
                               trconf.max_lr,
                               cooldown_lr))


# Generate a weight decay mask
# Exclude biases and layer norm /rms norm parameters
graphdef, params, _ = nnx.split(m, nnx.Param, nnx.Variable)
weight_decay_mask = jax.tree.map(lambda x: len(x.value.shape) > 1, params, 
                                 is_leaf=lambda n: isinstance(n, nnx.Param))

tx = optax.chain(
    optax.clip_by_global_norm(trconf.max_grad_norm),
    optax.adamw(trapezoidal_schedule, 
                b1=trconf.adam_b1, 
                b2=trconf.adam_b2, 
                weight_decay=trconf.weight_decay)
)
optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)

# count the number of weight decay params
def f(x, y):
    if x:
        return y.size
    return 0

weight_decay_params = jax.tree_util.tree_map(f, weight_decay_mask, params)
weight_decay_param_count = jax.tree_util.tree_reduce(lambda x, y: x + y, weight_decay_params, 0)
print(f"weight decay param count: {weight_decay_param_count:,}")
pprint(trconf)
print(f"effective batch size: {trconf.grad_accumulation_steps * trconf.mB}")
print(f"effective batch size per device: ", trconf.grad_accumulation_steps * trconf.mB // num_devices)


# Set up Dataloader

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./alpha-448101-282bc1b884cd.json"

train_dl = BlendedCloudDataLoader(
    device_rank=1,
    block_size=trconf.T,
    batch_size=trconf.mB,
    bucket_names=["jaxpt_datasets", "jaxpt_datasets", "jaxpt_datasets"],
    bucket_prefixes=["smollm-corpus/processed/fineweb-edu-dedup",
    "smollm-corpus/processed/python-edu",
    "smollm-corpus/processed/cosmopedia-v2"],
    proportions=[85, 1, 12],
    start_shards=[545, 7, 76],
    label="train"
)
jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32") 

output_dir = Path("/workspace/alpha_training_runs") # Lambda Labs setup
print(f"Output dir: {output_dir}")

timestamp = datetime.now().strftime("%Y%m%d")
random_code = generate_readable_code()

run_dirname = f"run_{timestamp}_{random_code}"
print(f"Run: {run_dirname}")

# set up logging 

log_dir = output_dir / m.config.name / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
print(f"Log directory: {log_dir}")

train_losses = []
append_to_csv(log_dir / f"{run_dirname}_train.csv", ["step", "lr", "loss", "time", "tokens_processed", "tokens_per_sec"])
print(f"Starting from step: {optimizer.step.value.item()}")
start = False


with mesh:
  data_sharding = NamedSharding(mesh, PartitionSpec("devices",))
  m.train(add_noise=True, aux_loss=True)
  try:
    while optimizer.step.value.item() < trconf.max_steps:
      step = optimizer.step.value.item()
      batch, target = train_dl()
      batch = jax.device_put(batch.squeeze(), data_sharding)
      target = jax.device_put(target.squeeze(), data_sharding)
      avg_loss, aux_loss = train_step(m, optimizer, batch, target)
      iter_time = time.time() - start
      if step % trconf.print_interval == 0:
        if not start:
          start = time.time()
          iter_time = -1
          sub_step_time = -1
          tokens_per_sec = -1
        else:
          iter_time = (time.time() - start) / trconf.print_interval
          sub_step_time = iter_time / trconf.grad_accumulation_steps
          tokens_per_sec = trconf.mB * trconf.T * trconf.grad_accumulation_steps / iter_time

        tokens_processed = (step+1) * trconf.grad_accumulation_steps * trconf.mB * trconf.T
        lr = trapezoidal_schedule(step)
        avg_loss = avg_loss.item()

        train_losses.append((step, avg_loss))
        append_to_csv(log_dir / f"{run_dirname}_train.csv", [step, lr, avg_loss, iter_time*1000, tokens_processed, tokens_per_sec])
        print(f"{step} | lr: {lr:0.4f} | "
              f"loss: {avg_loss:0.4f} | "
              f"aux_loss: {aux_loss:0.4f} | "
              f"time: {iter_time*1000:0.2f}ms | "
              f"tokens processed: {tokens_processed:,} | "
              f"tok/sec: {tokens_per_sec:,.2f}", end="\r")
        start = time.time()
      if step > 0 and step % trconf.eval_interval == 0:
        print("Evaluation TBD")
      if step > 0 and step % trconf.checkpoint_interval == 0:
        print(f"Saving checkpoint at step {step}")
        save_checkpoint(m, output_dir, run_dirname, step)
        save_optimizer_state(optimizer)
  except KeyboardInterrupt:
      print("Received KeyboardInterrupt. Exiting...")
  finally:
    plt.figure(figsize=(7, 5))
    plt.plot([x[0] for x in train_losses], [x[1] for x in train_losses], label="train loss")
    plt.yticks(ticks=np.arange(0, 12, 0.5))
    plt.grid()
    plt.legend()
    plt.savefig(log_dir / f"{run_dirname}.png", dpi=300, bbox_inches="tight", transparent=True)

    save_checkpoint(m, output_dir, run_dirname, optimizer.step.value.item())
    save_optimizer_state(m, optimizer)
    print("Done.")
