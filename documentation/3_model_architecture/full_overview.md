# IndicPrayog Model Architecture: Complete Overview

## 1. What we are building

IndicPrayog is a decoder language model for Hindi, Marathi, and English. We implement its components in PyTorch and initialize its weights from scratch. The purpose is to understand the complete path from tokenized text to language modeling, including attention, sparse experts, routing, optimization, and evaluation.

The current implementation has **502,903,424 learned parameters** and **54,243,968 active transformer parameters per token**. The approximate A50M name refers to that transformer count. The full vocabulary output projection also runs for each predicted token, bringing the active transformer plus output projection count to **103,395,968**.

The tokenizer and data preparation are complete. Architecture implementation and CPU correctness checks are complete for the reference implementation. Pretraining, GPU throughput measurements, and language quality evaluation remain planned work.

Read the [tokenizer overview](../1_tokenizer/full_overview.md) and [data processing overview](../2_data_processing/full_overview.md) for the earlier phases. The [model guide](../../model/Readme.md) contains installation instructions and a shorter API reference.

## 2. How the overall architecture works

![IndicPrayog architecture overview with attention, a dense layer, and sparse expert layers](../../assets/IndicPrayog_500M_A50M_architecture.png)

Read the main stack from the bottom upward. The tokenizer supplies integer IDs. An embedding table turns each ID into a vector. A stack of decoder layers transforms those vectors using causal attention and feed forward networks. Final normalization and a vocabulary projection produce the scores used to predict the next token.

The first decoder layer uses a dense feed forward network. The remaining eighteen decoder layers use a Mixture of Experts, abbreviated MoE. Every layer has attention, normalization, and residual connections. The repeated block in the diagram should therefore be read with this distinction: layer 0 is dense, and layers 1 to 18 are sparse.

For batch size B and sequence length S, the default tensor shapes are:

```text
Input IDs:                    [B, S]
Token embeddings:             [B, S, 768]
Dense decoder layer 0:        [B, S, 768]
Sparse decoder layers 1 to 18: [B, S, 768]
Final normalization:          [B, S, 768]
Vocabulary logits:            [B, S, 64000]
```

The diagram labels 128K as the supported context. In the implementation, 131,072 is the configured position limit. The prepared training sequences contain 2,048 tokens. Useful performance at longer contexts must be established through training and evaluation.

### Diagram to code map

<table>
<thead><tr><th>Diagram component</th><th>Implementation</th><th>What it does</th></tr></thead>
<tbody>
<tr><td>Token embedding</td><td><a href="../../model/modeling.py"><code>IndicPrayogForCausalLM.embed_tokens</code></a></td><td>Looks up one vector for each token ID.</td></tr>
<tr><td>Repeated decoder block</td><td><a href="../../model/modeling.py"><code>DecoderLayer</code></a></td><td>Combines attention, feed forward processing, normalization, and residual additions.</td></tr>
<tr><td>RMSNorm blocks</td><td><a href="../../model/layers.py"><code>RMSNorm</code></a></td><td>Normalizes vectors using an FP32 reduction and a learned scale.</td></tr>
<tr><td>Grouped query attention</td><td><a href="../../model/layers.py"><code>GroupedQueryAttention</code></a></td><td>Computes causal attention using shared key and value heads.</td></tr>
<tr><td>RoPE positions</td><td><a href="../../model/layers.py"><code>RotaryEmbedding</code></a></td><td>Rotates queries and keys according to their positions.</td></tr>
<tr><td>Dense feed forward block</td><td><a href="../../model/layers.py"><code>SwiGLU</code></a></td><td>Processes every token in the first layer.</td></tr>
<tr><td>Router</td><td><a href="../../model/moe.py"><code>TopKRouter</code></a></td><td>Selects experts, calculates mixture weights, and records assignment counts.</td></tr>
<tr><td>Routed and shared experts</td><td><a href="../../model/moe.py"><code>SparseMoE</code></a></td><td>Dispatches valid tokens and combines expert outputs.</td></tr>
<tr><td>Final normalization and output projection</td><td><a href="../../model/modeling.py"><code>IndicPrayogForCausalLM.norm</code> and <code>lm_head</code></a></td><td>Produces a score for every vocabulary entry.</td></tr>
</tbody>
</table>

## 3. How we adapted the Sarvam design

[Sarvam 30B](https://huggingface.co/sarvamai/sarvam-30b) provides the architectural reference. Its [published configuration](https://huggingface.co/sarvamai/sarvam-30b/blob/main/config.json) describes the dimensions used as our starting point. We retain the pattern of a leading dense layer followed by sparse layers, routed and shared experts, and grouped query attention. We train our own weights and use our own tokenizer and prepared data.

Our width multiplier is:

```text
Width multiplier: 768 / 4096 = 0.1875

Hidden width:     4096 × 0.1875 = 768
Dense FFN width:  8192 × 0.1875 = 1536
Expert FFN width: 1024 × 0.1875 = 192
```

A matrix whose two dimensions both shrink by this factor has approximately `0.1875² = 0.03515625` times as many weights. This describes the individual matrices. The complete model budget also depends on vocabulary size, expert count, attention grouping, and shared components.

We keep nineteen layers, choose forty six routed experts per sparse layer, and use a custom vocabulary of 64,000 entries for the three target languages. Each token selects two routed experts, while one shared expert is always available to every valid token. These choices bring the complete model close to the 0.5B storage target while keeping the active transformer close to the 50M target. The dimensions are an experimental design choice; matching a larger model's structure does not establish matching language quality.

## 4. Configuration and construction

[IndicPrayogConfig](../../model/config.py) stores the architecture settings. It is a frozen dataclass: an instantiated configuration cannot be edited in place. Validation checks integer dimensions, finite numeric values, supported activation and routing choices, attention head compatibility, and the even head dimension required by rotary positions.

Derived properties calculate the number of dense and sparse layers, stored and active experts, and query heads per key and value head. The model calls `is_moe_layer(layer_idx)` when constructing each `DecoderLayer`; this determines whether its `ffn` is a `SwiGLU` or a `SparseMoE`.

Configuration and analytical parameter counting work without importing PyTorch. The package's [public imports](../../model/__init__.py) load the model classes lazily when requested.

## 5. Normalization and residual connections

[RMSNorm.forward](../../model/layers.py) calculates a token vector's mean squared value in FP32, divides by the resulting root mean square, applies a learned scale, and returns the input dtype. The small epsilon prevents division by zero.

The decoder applies normalization before attention and before the feed forward network. Each branch then adds its result back to the residual stream:

```text
After attention: x + attention(input_norm(x))
Layer output:   x + ffn(post_attention_norm(x))
```

These expressions describe successive updates to x. The default model also applies normalization to query and key heads before rotary positions, and once more before the vocabulary projection. The `use_rmsnorm` and `use_qk_norm` options control the corresponding normalization modules.

## 6. Grouped query attention and positions

![Grouped query attention showing query, key, and value projections](../../assets/gqp_architecutre.png)

The attention diagram starts with hidden vectors of width 768. Queries use twelve heads; keys and values each use two heads. Each head has width 64, so every key and value head serves six query heads.

The drawing groups projection into one linear stage. The code uses three separate modules: `q_proj`, `k_proj`, and `v_proj`. Their output widths are 768, 128, and 128 respectively. Splitting those outputs into heads gives:

```text
Queries: [B, 12, S, 64]
Keys:    [B,  2, S, 64]
Values:  [B,  2, S, 64]
```

The implementation normalizes queries and keys, applies `RotaryEmbedding`, then repeats the key and value heads for the attention calculation. `torch.nn.functional.scaled_dot_product_attention` computes the weighted value mixture with the usual scaling by the square root of the head dimension. `o_proj` maps the combined heads back to width 768. Dropout is disabled during evaluation.

`RotaryEmbedding.forward` constructs inverse frequencies and angles in FP32 with autocast disabled, then casts the rotated queries and keys back to their input dtypes. Frequencies are computed on demand, so converting model weights to BF16 does not convert a stored frequency buffer to BF16.

### Causality, padding, and cache positions

[IndicPrayogForCausalLM.forward](../../model/modeling.py) prepares the attention mask before calling the decoder. For an ordinary unpadded sequence, it uses SDPA's causal mode. For padding or a cached chunk that needs an offset, it constructs an explicit causal mask.

The public `attention_mask` is a two dimensional tensor: 1 means a valid token, and 0 means padding. The model combines valid keys with causal structure. Padding positions are excluded from expert routing and from valid loss transitions. Position IDs are inferred from valid token counts when a padding mask is supplied.

Packed documents are separated by EOS tokens. Attention remains causal across document boundaries inside a packed row; the implementation does not reset attention at each EOS.

## 7. Dense processing and sparse experts

![Dense SwiGLU processing compared with routed and shared expert processing](../../assets/moe_routing_and_layer.png)

The lower panel describes layer 0. Its dense feed forward network processes every token with an intermediate width of 1,536. Both routed and shared experts in later layers use the same [SwiGLU](../../model/layers.py) building block with a smaller intermediate width of 192.

SwiGLU has three matrices:

```text
Gate branch: gate_proj(x), followed by SiLU
Value branch: up_proj(x)
Output: down_proj(SiLU(gate_proj(x)) × up_proj(x))
```

The upper panel shows the available expert paths. Only the two selected routed experts contribute for a token; the arrows do not mean that all forty six experts run for every token. The shared expert receives the token vector directly through `SparseMoE.shared_experts`. It does not depend on a router selection.

### Router selection and mixture weights

[TopKRouter.forward](../../model/moe.py) performs these operations:

1. Project each valid token to one logit per routed expert. The default router computation uses FP32 with autocast disabled.
2. Apply sigmoid to obtain independent expert scores. Softmax is an optional configuration choice.
3. Add the expert balancing bias for selection, then choose the highest scoring experts.
4. Gather the original scores for those experts, without adding the balancing bias to the mixture weights.
5. Normalize the selected scores when more than one expert is selected, then multiply by the routed scaling factor of 2.5.

The selection bias therefore changes which experts are chosen, while the mixture weights retain gradients through the original scores. For the optional single expert configuration, the score is retained without normalization so the gate can still receive a gradient.

### Dispatch and the shared path

[SparseMoE.forward](../../model/moe.py) flattens valid tokens, calls the router, and finds the token assignments for each expert. Experts without assignments are skipped. Selected expert outputs are weighted and accumulated back into token order. The shared expert output is added once, without the routed scaling factor.

There is no capacity limit and no assignment dropping. The implementation uses a Python expert loop and regular PyTorch operations. Its sparse parameter participation does not by itself establish a particular GPU throughput.

### Load balancing across optimizer steps

The router keeps two kinds of state. `expert_bias` is a persistent FP32 buffer, not a learned parameter. `token_counts` is a temporary integer buffer that accumulates assignments during training forwards and is excluded from saved checkpoints.

Call `IndicPrayogForCausalLM.update_router_bias()` once after a successful optimizer step, after all accumulated microbatches. Underloaded experts receive a bias increase and overloaded experts receive a decrease. The method then clears the pending counts. Evaluation forwards do not accumulate counts or update biases.

This implementation uses explicit bias balancing. `CausalLMOutput` returns `logits`, optional `loss`, and optional `past_key_values`; it does not return a router auxiliary loss to add to cross entropy. The router's configured dtype controls arithmetic, while parameter storage follows the model dtype. The bias buffer remains FP32 during dtype conversion.

For replicated distributed training, the model guide describes `find_unused_parameters=True`, `broadcast_buffers=False`, and a synchronized bias update on every rank. The update aggregates counts before changing biases. Actual execution across multiple GPUs remains unverified.

## 8. Exact parameter accounting

[parameter_counts](../../model/parameter_count.py) calculates the budget directly from the configuration. It separates learned parameters from balancing state:

```text
Input embeddings:                      49,152,000
Output projection:                     49,152,000
Attention projections across layers:   26,148,864
All normalization weights:                 32,384
Dense feed forward weights:              3,538,944
Router weights:                           635,904
Routed and shared expert weights:     374,243,328
Total learned parameters:             502,903,424
```

The normalization category includes the query and key norms. One expert has `3 × 768 × 192 = 442,368` weights. Eighteen sparse layers each store forty seven experts, giving `18 × 47 × 442,368 = 374,243,328` expert weights.

For one token, only three experts per sparse layer contribute. Their weights total `18 × 3 × 442,368 = 23,887,872`. Adding all always active transformer components gives the active transformer count below:

<table>
<thead><tr><th>Quantity</th><th>Count</th><th>Interpretation</th></tr></thead>
<tbody>
<tr><td>Total learned parameters</td><td>502,903,424</td><td>All embedding, output, transformer, router, and expert weights.</td></tr>
<tr><td>Active transformer parameters per token</td><td>54,243,968</td><td>Attention, norms, dense FFN, routers, and selected plus shared experts.</td></tr>
<tr><td>Active transformer plus output projection</td><td>103,395,968</td><td>Also includes the complete vocabulary projection used for prediction.</td></tr>
<tr><td>Embedding values accessed per token</td><td>768</td><td>One selected row from the input embedding table.</td></tr>
<tr><td>Expert balancing bias values</td><td>828</td><td>Persistent state buffers, excluded from learned parameter totals.</td></tr>
</tbody>
</table>

The earlier 502,904,252 estimate included the 828 balancing bias values as parameters. In the current implementation those values are buffers, giving 502,903,424 learned parameters. Parameter participation is separate from FLOPs, optimizer memory, and the union of experts selected by a complete batch.

The analytical budget is available without constructing the model:

```python
from model import IndicPrayogConfig
from model.parameter_count import parameter_counts

counts = parameter_counts(IndicPrayogConfig())
print(counts["total_parameters"])
print(counts["active_transformer_parameters_per_token"])
```

## 9. Initialization and training loss

[IndicPrayogForCausalLM._init_weights](../../model/modeling.py) initializes embedding and linear weights from a normal distribution with the configured standard deviation, 0.02 by default. Attention output and feed forward output projections use that standard deviation divided by `sqrt(2 × num_hidden_layers)`. RMSNorm scales start at one. Input embeddings and the output projection are separate by default; the configuration also supports tying them.

Training uses next token cross entropy. Supply the same unshifted token IDs as `input_ids` and `labels`. The model aligns each prediction with the following label internally. A 2,048 token row supplies 2,047 possible target predictions before ignored labels and padding are considered.

The label value negative 100 marks an ignored target. A padding mask excludes transitions involving padding. If no valid target remains, the model returns a differentiable zero loss. Logits are converted to FP32 for the loss calculation.

The following small configuration demonstrates an optimizer step without allocating the complete model. It illustrates the API; it does not define the final pretraining schedule.

```python
import torch
from model import IndicPrayogConfig, IndicPrayogForCausalLM

torch.manual_seed(42)
config = IndicPrayogConfig(
    vocab_size=128,
    hidden_size=64,
    num_hidden_layers=3,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    max_position_embeddings=128,
    intermediate_size=128,
    moe_intermediate_size=32,
    num_experts=4,
)
model = IndicPrayogForCausalLM(config)
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=0.0003)
input_ids = torch.randint(4, config.vocab_size, (1, 16))

optimizer.zero_grad(set_to_none=True)
output = model(input_ids, labels=input_ids)
output.loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
model.update_router_bias()
```

The full dataset loader must read the packed uint16 files and convert batches to integer tensors accepted by the embedding and loss. Using `torch.long` for IDs and labels is appropriate. The data overview reports 9,960,701,952 training tokens, with separate validation files. Validation tokens remain outside training.

## 10. Generation, caching, and saved state

[GroupedQueryAttention.forward](../../model/layers.py) can return a cache containing keys and values before head repetition. Each layer stores tensors shaped `[B, 2, cached_length, 64]`. Later inference calls reuse those keys and values rather than recomputing the prefix. Cached chunks receive the correct position and causal offsets.

[IndicPrayogForCausalLM.generate](../../model/modeling.py) supports greedy decoding and optional sampling with temperature and a vocabulary candidate limit. Batched prompts must be unpadded or use left padding with a valid mask. When passing a cache directly, the attention mask covers both cached and current tokens. Cache tensors must match the active attention device and dtype. Labels cannot be combined with caching.

Saving and loading use this package's own checkpoint format:

```python
from tempfile import TemporaryDirectory

model.eval()
generated_ids = model.generate(input_ids[:, :8], max_new_tokens=4, eos_token_id=1)

with TemporaryDirectory() as directory:
    model.save_pretrained(directory)
    restored = IndicPrayogForCausalLM.from_pretrained(directory)
    restored.eval()
```

`save_pretrained` writes `config.json` and `model.pt`, including persistent balancing biases. Temporary assignment counts are excluded. Save after router bias updates. A resumable trainer must separately save optimizer and scheduler state, random state, token progress, and data loader position. This format is a local model checkpoint rather than a Transformers export.

The model starts with random weights. A successful generation call validates the inference path; fluent output must be established after training.

## 11. Where each file fits

1. [config.py](../../model/config.py): architecture settings, validation, derived properties, and JSON serialization.
2. [layers.py](../../model/layers.py): normalization, rotary positions, grouped query attention, and SwiGLU.
3. [moe.py](../../model/moe.py): routing, selected expert dispatch, shared processing, and explicit balancing updates.
4. [modeling.py](../../model/modeling.py): decoder construction, residual flow, masks, language modeling loss, caching, generation, and persistence.
5. [parameter_count.py](../../model/parameter_count.py): analytical learned and active parameter budgets.
6. [smoke_test.py](../../model/smoke_test.py): executable checks for finite forward and backward computation with tiny or full dimensions.
7. [__init__.py](../../model/__init__.py): public configuration, model, and output imports.
8. [Readme.md](../../model/Readme.md): installation, verification commands, and concise usage guidance.

The current implementation groups the low level components in `layers.py`. Diagram labels describe computations; they are not separate filenames for every box.

## 12. What the checks establish

[Configuration tests](../../tests/test_config.py) cover defaults, validation, immutable settings, serialization, dependency free configuration imports, and the default parameter budget. [Model tests](../../tests/test_model.py) compare attention, normalization, rotary positions, sparse dispatch, and gradients with reference calculations.

The model suite also checks causal prefix invariance, padding behavior, valid target handling, cached inference, BF16 computation, generation, tied embeddings, and saved state. Router tests check that unused experts are skipped, balancing bias changes selection without changing the underlying mixture scores, and assignment counts accumulate correctly. The distributed balancing test mocks the count reduction; it is not a live test across multiple GPUs.

All 31 CPU tests and both tiny and full BF16 smoke checks passed during integration. The full check executes a short sequence through the actual default model and verifies its analytical and instantiated parameter counts. These checks establish implementation behavior, not trained quality or production throughput.

## 13. What remains before the training run

The reference model does not include a production training harness, fused expert kernels, or activation checkpointing. A trainer still needs a shuffled packed data loader, a measured batch size, an optimizer schedule, validation, logging, and interruption recovery. Router statistics need special care if activation recomputation is added, so the same tokens are not counted twice.

The full vocabulary logits and FP32 loss can consume substantial memory. Select the batch size using measured GPU peak memory through an optimizer step. The configured maximum position count and sparse parameter count do not establish a training speed, cost, or context quality guarantee.

The next phase is to verify the full training loop on the intended hardware, then pretrain and evaluate language modeling and generation separately for Hindi, Marathi, and English. The [project README](../../README.md) tracks the overall goal and progress.
