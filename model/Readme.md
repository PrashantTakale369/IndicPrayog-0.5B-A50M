# Model

IndicPrayog-0.5B-A50M architecture: a scaled down Sarvam-30B style Mixture of Experts model. 19 layers where layer 0 uses a dense feed forward network and layers 1 to 18 are Mixture of Experts, grouped query attention, RoPE positions, SwiGLU experts, 2 routed experts plus 1 shared expert per token. 502.9M total parameters, 54.2M active per token.

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