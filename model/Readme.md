# IndicPrayog sparse MoE architecture

A decoder-only language model implemented from scratch with PyTorch modules. No
pretrained model weights or Transformers model classes are used. The dimensions
are inspired by [Sarvam-30B's published configuration](https://huggingface.co/sarvamai/sarvam-30b/blob/main/config.json);
this is an independent implementation, not a checkpoint-compatible Sarvam port.

For a diagram by diagram explanation and code mapping, read the [model architecture overview](../documentation/3_model_architecture/full_overview.md).

## Architecture

| Component | Default |
|---|---|
| Vocabulary / hidden size | 64,000 / 768 |
| Decoder layers | 19: layer 0 dense, layers 1–18 sparse MoE |
| Attention | Causal GQA, 12 query heads, 2 KV heads, dimension 64 |
| Normalization | Pre-RMSNorm, per-head Q/K RMSNorm, final RMSNorm |
| Positions | RoPE, theta 8,000,000, configured limit 131,072 |
| Dense FFN | SwiGLU, intermediate dimension 1,536 |
| Each MoE layer | 46 routed experts and 1 shared expert, dimension 192 each |
| Per token, per MoE layer | 2 routed experts selected + 1 shared expert |
| Router | fp32 sigmoid scores, top-2 selection, normalized selected weights × 2.5 |
| Load balancing | Selection-only expert bias, updated explicitly after an optimizer step |
| Embeddings / output head | Untied by default |
| Packed training context | 2,048 tokens |

The configured RoPE limit is not evidence of trained long-context capability. The
dataset uses 2,048-token rows. EOS separates packed documents; causal attention
continues across document boundaries within each row.

## Parameter accounting

Run `python -m model.parameter_count` without PyTorch installed. Counts are also
checked against a full model instantiated on the meta device in the test suite.

| Quantity | Parameters |
|---|---:|
| Total learned parameters | 502,903,424 |
| Active transformer parameters per token | 54,243,968 |
| Active transformer + full vocabulary output projection | 103,395,968 |
| Input embedding matrix | 49,152,000 |
| Output projection matrix | 49,152,000 |
| Expert balancing bias values (buffers, excluded from learned totals) | 828 |

The approximately 50M activation target refers to the transformer, including
attention, norms, dense FFN, every router, and selected/shared experts. The full
output projection also runs on every predicted token. The input embedding is a
768-value lookup per token. These counts describe parameter participation, not
FLOPs, memory bandwidth, or the union of experts selected by an entire batch.

The earlier config-only estimate counted balancing biases as learned parameters;
storing them correctly as buffers reduces that estimate by 828.

## Files

| File | Responsibility |
|---|---|
| `config.py` | Immutable, validated config; JSON serialization; stdlib only |
| `layers.py` | RMSNorm, RoPE, causal GQA and SwiGLU |
| `moe.py` | Sparse dispatch, shared experts, routing and bias updates |
| `modeling.py` | Residual decoder, LM head, loss, KV cache, generation, save/load |
| `parameter_count.py` | Exact analytical counts |
| `smoke_test.py` | Executable forward/backward check, tiny or full model |
| `../tests/` | Numerical and behavioral correctness tests |

## Install and verify

Use Python 3.10 or later from the repository root. For a CPU environment:

```bash
python -m pip install -r requirements-model.txt --index-url https://download.pytorch.org/whl/cpu
python -m unittest discover -s tests -v
python -m model.smoke_test
python -m model.smoke_test --full
```

For CUDA, install the appropriate PyTorch build for your GPU first, then run
`python -m model.smoke_test --full --device cuda`. The default smoke dtype is
bfloat16; `--dtype float32` is also available. The full model needs approximately
1 GB for bfloat16 weights alone, plus gradients, activations, and runtime memory.

## Forward and training integration

```python
import torch
from model import IndicPrayogConfig, IndicPrayogForCausalLM

model = IndicPrayogForCausalLM(IndicPrayogConfig())
model.train()
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)

# Illustrative IDs. Real inputs come from the packed uint16 dataset, converted
# to torch.long on the model's device. Set batch size to fit available memory.
input_ids = torch.randint(4, model.config.vocab_size, (1, 2048))
optimizer.zero_grad(set_to_none=True)
output = model(input_ids, labels=input_ids)
output.loss.backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
model.update_router_bias(update_rate=1e-3)
```

Labels are **unshifted**: logits at position `t` predict labels at `t + 1`.
Use `-100` for ignored labels. A 2D `attention_mask` uses 1 for valid tokens and
0 for padding; pad transitions are excluded from loss and padding does not count
toward expert load. A batch with no valid prediction targets returns zero loss.

Expert dispatch is dropless: only selected experts run, and every assignment is
retained. The shared experts always process valid tokens. Bias changes selection,
not mixture weights. Sigmoid or softmax routing is configurable; selected scores
are normalized when `top_k > 1`. Top-1 retains its score so its gate still receives
gradients. Router computation stays fp32 under autocast by default, and bias
buffers stay fp32 when the model is converted to bfloat16.

Call `update_router_bias` **once per optimizer step**, after all accumulated
microbatches. Counts are accumulated during training forwards and reset by this
call. Evaluation forwards do not change router counts or bias. If using AMP with
skipped optimizer steps, update bias only when the optimizer actually steps and
clear pending `router.token_counts` when discarding a step's microbatches.

For DDP, use `find_unused_parameters=True` because some experts can receive no
tokens, and `broadcast_buffers=False` to preserve each rank's accumulated routing
counts. Every rank must call `update_router_bias(synchronize=True)` together; it
sums counts across ranks before updating. This is replicated DDP, not expert
parallelism. Multi-GPU execution has not been validated in this environment.

This reference implementation uses Python expert dispatch and ordinary PyTorch
SDPA. Padding/cached chunks use explicit masks; the unpadded training path uses
SDPA's causal mode. The cache stores only two KV heads per layer. See the
[PyTorch SDPA mask documentation](https://docs.pytorch.org/docs/stable/generated/torch.nn.functional.scaled_dot_product_attention.html)
for mask semantics. There are no fused expert kernels, gradient checkpointing,
or production training harness in this change. Gradient checkpointing would need
to avoid double-counting router statistics during recomputation.

## Inference and persistence

```python
model.eval()
with torch.no_grad():
    prefix = model(input_ids[:, :16], use_cache=True)
    continuation = model(
        input_ids[:, 16:17], past_key_values=prefix.past_key_values, use_cache=True
    )

generated = model.generate(input_ids[:, :16], max_new_tokens=20, eos_token_id=1)
model.save_pretrained("checkpoints/example")
restored = IndicPrayogForCausalLM.from_pretrained("checkpoints/example")
```

Generation supports greedy decoding or `do_sample=True` with temperature and
optional `top_k`. Batched prompts must be unpadded or left-padded with an explicit
mask. Caches must stay on the same device and attention dtype between calls. With
padding and a cache, supply a mask covering both the prefix and new tokens.

`save_pretrained` writes this package's `config.json` and `model.pt`; it is not a
Hugging Face Transformers export. Save after bias updates: pending microbatch
counts are transient and excluded from checkpoints. Optimizer, scheduler, RNG,
and data-loader state need separate handling for a resumable training run.
The model starts with random weights; meaningful generation requires training.
