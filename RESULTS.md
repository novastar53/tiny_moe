# UNet Skip Connection Experiment Results

## Summary

We investigated whether UNet-style skip connections could improve Tiny_MoE's performance, inspired by their success in modded-nanogpt. **Result: Skip connections hurt performance** - loss was consistently higher throughout training compared to baseline.

## Implementation Details

### Architecture

We implemented trainable gated skip connections following the modded-nanogpt pattern:

- **`skip_lambda`**: Global learnable scalar controlling overall skip strength
- **`skip_gate`**: Small linear layer (12 input features → 1) for position-wise gating
- **Connection pattern**: Layers 0-14 skip to layers 15-29 (LIFO: layer 0 → 29, layer 1 → 28, etc.)

### Code Location

The skip connection logic was added to `tiny_moe.py` in the `Tiny_MoE` class:
- Skip activations stored during forward pass through early layers
- Gated residual added to corresponding late layers
- Both weighted (with learned gates) and unweighted variants tested

## Results

| Configuration | Outcome |
|---------------|---------|
| Baseline (no skips) | Best performance |
| Unweighted skip connections | Higher loss than baseline |
| Gated skip connections | Higher loss than baseline |

The skip connections consistently degraded model performance regardless of gating strategy.

## Analysis: Why Skip Connections Work in modded-nanogpt But Not Here

### The Critical Difference

| Aspect | modded-nanogpt | Tiny_MoE |
|--------|----------------|----------|
| Number of skips | **1** (layer 3 → 6) | **15** (layers 0-14 → 15-29) |
| Purpose | Fill architectural gap | General information preservation |
| Layer 6 attention | **Removed** | Present |
| Skip source | Long-window attention layer | All early layers |

### Key Insight

In modded-nanogpt, the skip connection serves a **specific architectural purpose**:

1. **Layer 6 has NO attention** - only MLP. This is a deliberate architectural choice for efficiency.
2. **Layer 3 uses long-window attention** - it captures broad contextual information.
3. **The skip connection compensates** for layer 6's missing attention by providing contextual information from layer 3.

Without this skip, layer 6 would be "blind" to token interactions - it would only see the residual stream without any new attention-based mixing at that layer.

### Why It Fails in Tiny_MoE

In Tiny_MoE, **every layer has full attention + MoE**. There is no architectural gap to fill:

- Each layer already has access to the full context via attention
- Each layer already has powerful nonlinear transformation via MoE
- Skip connections add redundant (and potentially noisy) signal
- The model must learn to ignore or compensate for this extra information

The skip connections in modded-nanogpt are a **surgical fix for a specific architectural limitation**, not a general-purpose improvement technique.

## Conclusion

Skip connections are not universally beneficial for transformers. Their effectiveness depends on the specific architecture:

- **Use when**: There's an architectural gap (e.g., attention-free layers that need contextual information)
- **Avoid when**: All layers already have full representational capacity

For Tiny_MoE, the standard transformer architecture with attention + MoE at every layer is already sufficient. Adding skip connections introduces unnecessary complexity without benefit.

## Recommendations

1. **Do not add skip connections to Tiny_MoE** - the current architecture is more effective
2. **Consider skip connections only if** removing attention from certain layers for efficiency gains
3. **If exploring efficiency optimizations**, the modded-nanogpt approach of removing attention + adding targeted skips could be investigated as a package deal
