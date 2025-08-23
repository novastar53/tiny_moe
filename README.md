# Tiny MoE

<img src="assets/0_1.png" style="width: 50%; height: auto;" />

## Overview

Tiny MoE is a minimal implementation of a Mixture-of-Experts (MoE) language model using JAX and Flax NNX. It demonstrates how to build an efficient and scalable language model that uses expert routing to process tokens.

## Features

- Minimal and clean implementation of Mixture-of-Experts architecture
- Built with JAX and Flax NNX for efficient computation
- Use GLU (Gated Linear Units) and GLU based MoE blocks
- Implements RoPE (Rotary Position Embedding) for better position encoding
- Uses model parallelism for the Expert layers
- Uses data parallelism for the non-MoE layers
- Auxiliary loss for load balancing between experts

## Requirements

- Python 3.13.1
- Core dependencies:
  - JAX (with CUDA 12 or Metal support)
  - Flax
  - Orbax
  - Transformers

## Installation

1. Clone the repository:
```bash
git clone https://github.com/novastar53/tiny_moe.git
cd tiny_moe
```

2. Install the required dependencies:

First, install the `uv` package manager (if not already installed):
```bash
make uv
```

Then, install dependencies based on your hardware:

For CUDA support (NVIDIA GPUs):
```bash
make cuda
```

For CPU or Apple Metal support:
```bash
make cpu
```

The above commands will install all required dependencies including development tools.

## Model Architecture

The model consists of alternating MOE and GLU blocks:

- **MOE Block**: Combines attention mechanism with mixture-of-experts routing
- **GLU Block**: Uses gated linear units for non-linear transformations
- **Attention**: Implements multi-head attention with RoPE positional embeddings
- **RMSNorm**: Used for layer normalization


## Training

The repository includes scripts for training and evaluation:

- `train.py`: Main training script
- `eval.py`: Evaluation script
- `generate.py`: Text generation script

Example training command:
```bash
python train.py
```

## Dataset

The repository includes a sample dataset (Panchatantra stories) for testing and demonstration purposes. You can replace it with your own dataset by following the format in `dataloader.py`.

## License

This project is licensed under the GNU General Public License v3.0 - see the [LICENSE](LICENSE) file for details.

## Citation

If you use this code in your research, please cite it as:

```bibtex
@software{tiny_moe2025,
  author = {Vikram Pawar},
  title = {Tiny MoE: A Minimal Mixture-of-Experts Language Model Implementation},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/novastar53/tiny_moe}
}
```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgments

This implementation draws inspiration from and builds upon the work of several key projects:

- [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- [Huggingface SmolLM](https://huggingface.co/blogs/smollm)
- [Meta Mobile LLM](https://github.com/facebookresearch/mobile-llm)
- [Meta LLaMA](https://github.com/facebookresearch/llama)
- [Google DeepMind VMOE](https://github.com/google-deepmind/vmoe)

Additional acknowledgments:
- The JAX and Flax community for their excellent tools and support
- The authors of the original MoE papers that laid the groundwork for this field