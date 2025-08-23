#!/usr/bin/env python
# coding: utf-8
# Let's Train Tiny MoE
# Author: Vikram Pawar (pvikram035 [at] gmail [dot] com)

import os

#os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "./alpha-448101-282bc1b884cd.json"

import time
from functools import partial
from datetime import datetime
from pathlib import Path
from pprint import pformat
from dataclasses import dataclass

from matplotlib import pyplot as plt

import numpy as np
import jax
from jax.sharding import Mesh, PartitionSpec, NamedSharding
import jax.numpy as jnp
import flax
import flax.nnx as nnx
import optax


import tiktoken
from transformers import AutoTokenizer

from logging_config import setup_logging
from tiny_moe import Tiny_MoE, Config
from generate import generate
from dataloader import BlendedCloudDataLoader
from utils import (
    generate_readable_code, count_params, create_sharded_model, 
    save_checkpoint, save_optimizer_state, step_fn, append_to_csv
)

# Set up logging 
output_dir = Path("/workspace/training_runs").absolute()
timestamp = datetime.now().strftime("%Y%m%d")
random_code = generate_readable_code()
run_name = f"run_{timestamp}_{random_code}"
log_dir = output_dir / "Tiny_MoE" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)
train_logger = setup_logging(log_dir, f"output_{run_name}")
train_logger.info(f"Run: {run_name}")
train_logger.info(f"Log directory: {log_dir}")
train_logger.info(f"Output dir: {output_dir}")

jax.print_environment_info()
train_logger.info(f"Flax version: {flax.__version__}")
train_logger.info(f"Optax version: {optax.__version__}")
device = jax.default_backend()
train_logger.info(f"Platform: {device}")
devices = jax.devices()
num_devices = len(jax.devices())
train_logger.info(f"Num Devices: {num_devices}")
train_logger.info(f"Devices: {jax.devices()}")


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
jax.config.update("jax_default_matmul_precision", "BF16_BF16_F32") 

# Create model

rngs = nnx.Rngs(default=jax.random.key(1337), 
                gate_noise=jax.random.key(42))

config = Config(
            name="Tiny_MoE",
            dtype=jnp.bfloat16, \
            vocab_size=49152,
            n_layer=30,
            block_size=2048,
            n_head=9,
            n_kv_head=3,
            n_embed=576,
            n_glu_hidden=1536,
            expert_load_factor=1.25,
            sdpa_implementation="cudnn" if device=="gpu" else "xla")
train_logger.info(f"Model config:\n{pformat(config)}")

mesh = jax.sharding.Mesh(jax.devices(), ["devices"])
m = create_sharded_model(config, mesh, rngs)
graphdef, rngstate, state = nnx.split(m, nnx.RngState, ...)
total_params = count_params(m)
moe_params = count_params(m, "moe")

train_logger.info(f"Parameter Count: {total_params:,}")
train_logger.info(f"Sharded / MoE Parameter Count: {moe_params:,}")
train_logger.info(f"Replicated Parameter Count: {total_params - moe_params:,}")

# Set up training config

@dataclass
class TrainerConfig:
  num_tokens: int =  int(236e9)
  num_tokens_per_batch: int = 2**20 # 2**20 = 1.0 million
  mB: int = 64 * num_devices
  T: int = 2048
  max_steps: int = int(num_tokens // num_tokens_per_batch)
  max_lr: float = 6e-4
  min_lr: float = max_lr * 0.1
  max_grad_norm: float = 1.0  # Clip gradients to this norm
  weight_decay: float = 0.1 # Weight decay for adamw
  adam_b1: float = 0.9
  adam_b2: float = 0.95
  warmup_steps: int = max_steps // 100
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
                     warmup_lr,
                     jnp.where(step < 0.8 * trconf.max_steps,
                               trconf.max_lr,
                               cooldown_lr))

get_lr = partial(trapezoidal_schedule, trconf)

# Generate a weight decay mask
# Exclude biases and layer norm /rms norm parameters
graphdef, params, _ = nnx.split(m, nnx.Param, nnx.Variable)
weight_decay_mask = jax.tree.map(lambda x: len(x.value.shape) > 1, params, 
                                 is_leaf=lambda n: isinstance(n, nnx.Param))

tx = optax.chain(
    optax.clip_by_global_norm(trconf.max_grad_norm),
    optax.adamw(get_lr,
                b1=trconf.adam_b1,
                b2=trconf.adam_b2,
                weight_decay=trconf.weight_decay,
                mask=weight_decay_mask)
)
optimizer = nnx.Optimizer(m, tx, wrt=nnx.Param)

# Count the number of weight decay params
def f(x, y):
    if x:
        return y.size
    return 0

weight_decay_params = jax.tree_util.tree_map(f, weight_decay_mask, params)
weight_decay_param_count = jax.tree_util.tree_reduce(lambda x, y: x + y, weight_decay_params, 0)
train_logger.info(f"Weight decay param count: {weight_decay_param_count:,}")
train_logger.info(f"Training config:\n{pformat(trconf)}")
train_logger.info(f"Effective batch size: {trconf.grad_accumulation_steps * trconf.mB}")
train_logger.info(f"Effective batch size per device: {trconf.grad_accumulation_steps * trconf.mB // num_devices}")
assert(trconf.grad_accumulation_steps == 1)

# Set up Dataloader


train_dl = BlendedCloudDataLoader(
    device_rank=1,
    block_size=trconf.T,
    batch_size=trconf.mB,
    bucket_names=["jaxpt_datasets", "jaxpt_datasets", "jaxpt_datasets"],
    bucket_prefixes=["smollm-corpus/processed/fineweb-edu-dedup",
    "smollm-corpus/processed/python-edu",
    "smollm-corpus/processed/cosmopedia-v2"],
    proportions=[85, 1, 12],
    label="train"
)

# Train

with mesh:
    train_losses = []
    append_to_csv(log_dir / f"{run_name}_train.csv", ["step", "lr", "loss", "aux_loss", "time", "tokens_processed", "tokens_per_sec"])
    train_logger.info(f"Starting from step: {optimizer.step.value.item()}")
    start = False
    data_sharding = NamedSharding(mesh, PartitionSpec("devices",))
    m.train(add_noise=True, aux_loss=True)
    try:
        while optimizer.step.value.item() < trconf.max_steps:
            step = optimizer.step.value.item()
            batch, target = train_dl()
            batch = jax.device_put(batch.squeeze(), data_sharding)
            target = jax.device_put(target.squeeze(), data_sharding)
            avg_loss, aux_loss = step_fn(m, optimizer, batch, target)
            if step % trconf.print_interval == 0:
                if start is False:
                    start = time.time()
                    iter_time = 0 
                    tokens_per_sec = 0 
                else:
                    total_time = (time.time() - start)
                    iter_time = total_time / trconf.print_interval
                    tokens_per_sec = trconf.print_interval * trconf.mB * trconf.T / total_time

                tokens_processed = (step+1) * trconf.mB * trconf.T 
                lr = get_lr(step)
                avg_loss = avg_loss.item()

                train_losses.append((step, avg_loss))
                append_to_csv(log_dir / f"{run_name}_train.csv", [step, lr, avg_loss, aux_loss.item(), iter_time*1000, tokens_processed, tokens_per_sec])
                train_logger.info(f"{step} | lr: {lr:0.4f} | "
                        f"loss: {avg_loss:0.4f} | "
                        f"aux_loss: {aux_loss.item():0.4f} | "
                        f"avg iter time: {iter_time*1000:0.2f}ms | "
                        f"avg tok/sec: {tokens_per_sec:,.2f} | "
                        f"tokens processed: {tokens_processed:,}")
                start = time.time()
            if step > 0 and step % trconf.eval_interval == 0:
                train_logger.info("Evaluation TBD")
            if step > 0 and step % trconf.checkpoint_interval == 0:
                train_logger.info(f"Saving checkpoint at step {step}")
                save_checkpoint(m, output_dir, run_name, step)
                save_optimizer_state(m, output_dir, run_name, optimizer)
    except KeyboardInterrupt:
        train_logger.warning("Received KeyboardInterrupt. Exiting...")
    finally:
        plt.figure(figsize=(7, 5))
        plt.plot([x[0] for x in train_losses], [x[1] for x in train_losses], label="train loss")
        plt.yticks(ticks=np.arange(0, 12, 0.5))
        plt.grid()
        plt.legend()
        plt.savefig(log_dir / f"{run_name}_train.png", dpi=300, bbox_inches="tight", transparent=True)

        save_checkpoint(m, output_dir, run_name, optimizer.step.value.item())
        save_optimizer_state(m, output_dir, run_name, optimizer)
        train_logger.info("Training completed.")
