# Model

IndicPrayog-0.5B-A50M architecture: a scaled down Sarvam-30B style Mixture of Experts model. 19 layers where layer 0 uses a dense feed forward network and layers 1 to 18 are Mixture of Experts, grouped query attention, RoPE positions, SwiGLU experts, 2 routed experts plus 1 shared expert per token. Exactly 502,904,252 total parameters (502.9M) and 54,244,796 active per token (54.2M), both verified by instantiating the real model and counting.

## Step 1: config.py

`IndicPrayogConfig` is the single source of truth for every architectural number in the model. No other file in this package is allowed to hardcode a dimension, all of them read it from here. It depends on the standard library only, so the config stays importable and checkable before torch is installed.

The config validates itself the moment it is constructed. An impossible model is rejected here with a clear message instead of surfacing later as a shape mismatch deep inside a forward pass. Checks include head count times head dim equalling hidden size, query heads dividing evenly across key value heads, at least one Mixture of Experts layer remaining, routed expert count above the top k value, and vocabulary size staying inside the `uint16` limit the packed dataset relies on.

### Derived properties

These are computed from the base fields rather than stored, so there is no second copy of a number that can drift out of sync with the first.

| property | value with default config | what it means |
|----------|---------------------------|---------------|
| `num_dense_layers` | 1 | how many leading layers use a dense feed forward network instead of Mixture of Experts, taken straight from `first_k_dense_replace` |
| `num_moe_layers` | 18 | how many layers use Mixture of Experts, meaning total layers minus the dense ones |
| `experts_per_moe_layer` | 47 | all experts a single Mixture of Experts layer stores, that is 46 routed plus 1 shared |
| `active_experts_per_token` | 3 | experts one token actually passes through in a layer, that is 2 routed selected by the router plus the 1 shared expert every token always uses |
| `queries_per_kv_head` | 6 | how many query heads share one key value head under grouped query attention, that is 12 divided by 2 |

`is_moe_layer(layer_idx)` answers the same question per layer: it returns false for layer 0 and true for layers 1 to 18, and raises if the index falls outside the layer range.

## Step 2: norm.py

`RMSNorm` scales each token vector by its own root mean square and applies a learned per-dimension weight. The mean of squares is accumulated in fp32 even when the model runs in bf16, because the reduction loses too much precision in half precision, then the result is cast back to the input dtype. The same module is used four ways in the model: the input and post-attention layernorms of every layer, the final norm before the lm head, and the QK-norm inside attention.

## Step 3: rope.py

`RotaryEmbedding` precomputes the inverse frequencies from `rope_theta` and returns cosine and sine tables for a batch of position ids. Positions are encoded with the half-split convention: `rotate_half` pairs dimension i with dimension i + head_dim/2, and the cosine table is `cat(angles, angles)` to match. `apply_rotary_pos_emb` unsqueezes the tables so they broadcast across attention heads, and casts them to the query dtype. The frequency buffer is non-persistent, so it never appears in the state dict and is rebuilt on load. Head dim must be even, checked at construction.

## Step 4: attention.py

Grouped query attention: 12 query heads against 2 key value heads, with each key value head repeated 6 times via `repeat_kv` to serve its group. QK-norm normalizes each head's query and key vectors before RoPE is applied. The order matters: RoPE preserves vector length, so the learned per-dim scale of the norm must be applied in the unrotated frame.

The attention itself is `F.scaled_dot_product_attention`, which applies the 1/sqrt(head_dim) scale internally. When no mask is passed, `is_causal=True` builds the causal mask inside the kernel, which is the path used for packed training data. If a mask is passed it is handed to the kernel as is, so the caller must supply a pre-expanded 3D or 4D mask that already contains the causal structure, a plain 2D padding mask will not do the right thing. Dropout is active only in training mode.

## Step 5: experts.py

`SwiGLU` is the single feed forward building block: `down_proj(silu(gate_proj(x)) * up_proj(x))`, no biases anywhere. Two factory functions size it: `build_expert` uses `moe_intermediate_size` (192) for every routed and shared expert, and `build_dense_ffn` uses `intermediate_size` (1536) for layer 0's dense feed forward network.

## Step 6: router.py

The router computes per-expert affinity scores with a sigmoid (a softmax option exists), then selects the top-k experts per token. Selection and gating are deliberately separated, following the DeepSeek-V3 design: a learned-free expert bias is added to the scores only for top-k selection, while the gate values are the raw affinities of the selected experts renormalized to sum to 1. The router always runs in fp32 with autocast disabled, since routing decisions are too sensitive for half precision.

Two load balancing signals come out of the forward pass. The auxiliary loss follows the Switch Transformer form, `num_experts * sum(load_fraction * mean_score)`, which equals 1.0 under perfectly uniform load and grows as load concentrates; it is differentiable through the mean scores. The load fraction itself is the hard per-expert share of assignments and is consumed by `update_expert_bias`, which moves each bias by `update_rate * sign(1/num_experts - load_fraction)`. The bias parameter has `requires_grad=False`: backprop never touches it, only this explicit update does.

## Step 7: moe.py

`MoELayer` flattens the batch into tokens, routes them, then groups the flattened token-expert assignments by expert id using argsort and bincount, so each expert runs one batched matmul on its slice instead of 46 separate masking passes. Weighted expert outputs are accumulated back per token with `index_add_`, which is differentiable, so gradients reach both the experts and, through the gate weights, the router. The routed sum is multiplied by `routed_scaling_factor` (2.5) once, and the shared expert output is added unscaled on top. The layer returns the hidden states plus the router's auxiliary loss and load fraction.

The dispatch was verified bit-exact against a naive per-token top-k loop.

## Step 8: model.py

`DecoderLayer` is a standard pre-norm block: norm, attention, residual add, norm, feed forward (dense or MoE), residual add. MoE layers additionally return their auxiliary loss and load fraction. `IndicPrayogModel` stacks 19 of these, applies the final norm, sums the 18 auxiliary losses into one scalar, and stacks the load fractions into an [18, 46] tensor. `IndicPrayogForCausalLM` adds the untied lm head.

Weights initialize as N(0, 0.02), with the residual output projections (`o_proj`, `down_proj`) scaled down by 1/sqrt(2 * num_layers) so activations do not grow with depth. The training loss is next-token cross-entropy computed as logits[:-1] against labels[1:] with `ignore_index=-100`. The auxiliary loss is returned separately and unweighted: the trainer owns the combination, typically `loss = out.loss + aux_weight * out.aux_loss`, and drives the expert bias with `model.update_expert_bias(out.expert_load.detach())`.

`expected_total_parameters` and `expected_active_parameters` compute the parameter budget from the config alone. Both were verified against the instantiated model: the formula matches `count_parameters` exactly.

| component | parameters |
|-----------|------------|
| embedding + lm head | 98,304,000 |
| attention, 19 layers | 26,151,296 |
| dense feed forward, layer 0 | 3,538,944 |
| Mixture of Experts, 18 layers | 374,880,060 |
| all norms | 29,952 |
| **total** | **502,904,252** |

Active per token excludes embedding and lm head and counts only the experts a token passes through: 54,244,796.

## Usage

```python
import torch
from model import IndicPrayogConfig, IndicPrayogForCausalLM

config = IndicPrayogConfig()
model = IndicPrayogForCausalLM(config)

input_ids = torch.randint(0, config.vocab_size, (1, 32))
out = model(input_ids, labels=input_ids)

loss = out.loss + 0.01 * out.aux_loss
loss.backward()
model.update_expert_bias(out.expert_load.detach())
```

## Verification

The package was checked by executing every module against an independent reference implementation: attention against a hand-written GQA plus QK-norm plus causal mask reference, RoPE against explicit per-position rotation matrices, MoE dispatch against a naive per-token loop (bit-exact), and the loss against manual shifted cross-entropy. On the full 0.5B model the initial loss is 11.22 against the theoretical ln(64000) = 11.07 for a healthy random init, every non-expert parameter receives a finite gradient, and unused routed experts and `expert_bias` are the only parameters without gradients, both by design.
