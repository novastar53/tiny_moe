#!/usr/bin/env python
# coding: utf-8
# Let's Train Tiny MoE
# Author: Vikram Pawar (pvikram035 [at] gmail [dot] com)

import os

os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=8"
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
    count_params, 
    load_checkpoint
)

# set up logging 
output_dir = Path("training_runs").absolute()
run_name = "run_20250822_czechic_hovel"
log_dir = output_dir / "Tiny_MoE" / "logs"
eval_logger = setup_logging(log_dir, f"output_{run_name}")
eval_logger.info(f"Run: {run_name}")
eval_logger.info(f"Log directory: {log_dir}")
eval_logger.info(f"Output dir: {output_dir}")

jax.print_environment_info()
eval_logger.info(f"Flax version: {flax.__version__}")
eval_logger.info(f"Optax version: {optax.__version__}")
device = jax.default_backend()
eval_logger.info(f"Platform: {device}")
devices = jax.devices()
num_devices = len(jax.devices())
eval_logger.info(f"Num Devices: {num_devices}")
eval_logger.info(f"Devices: {jax.devices()}")


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
eval_logger.info(f"Model config:\n{pformat(config)}")

mesh = jax.sharding.Mesh(jax.devices(), ["devices"])
#m = create_sharded_model(config, mesh, rngs)
with mesh:
    m = load_checkpoint(output_dir, config, run_name, 95, rngs)

graphdef, rngstate, state = nnx.split(m, nnx.RngState, ...)
total_params = count_params(m)
moe_params = count_params(m, "moe")

eval_logger.info(f"Parameter Count: {total_params:,}")
eval_logger.info(f"Sharded / MoE Parameter Count: {moe_params:,}")
eval_logger.info(f"Replicated Parameter Count: {total_params - moe_params:,}")

#plt.figure(figsize=(7, 5))
#plt.plot([x[0] for x in train_losses], [x[1] for x in train_losses], label="train loss")
#plt.yticks(ticks=np.arange(0, 12, 0.5))
#plt.grid()
#plt.legend()
#plt.show()

with mesh:
    tokenizer = AutoTokenizer.from_pretrained("HuggingFaceTB/SmolLM-135M")
    x = tokenizer.encode("A wise king")
    x = generate(m, x, 10, 0.1, jax.random.key(1337))
    for i in range(x.shape[0]):
        print(tokenizer.decode(x[i]))
