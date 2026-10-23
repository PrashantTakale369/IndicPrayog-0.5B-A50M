# IndicPrayog-0.5B-A50M

An experimental 0.5B parameter sparse Indic language model with approximately 50M active parameters per token, trained from scratch using a scaled down Sarvam-30B inspired architecture for learning purpose only.

## Abstract

IndicPrayog is a Mixture of Experts language model for Hindi, Marathi, and English, built end to end in the open: custom tokenizer, own dataset pipeline, own architecture code, own training run. Nothing is inherited from a pretrained checkpoint.

The model has 502.9M total parameters of which only 54.2M are active per token, achieved with 19 layers where layer 0 uses a dense feed forward network and layers 1 to 18 each hold 46 routed experts plus 1 shared expert, with 2 routed experts selected per token. Attention is grouped query attention with 12 query heads over 2 key value heads, plus RoPE positions and QK normalization.

Training data is a 10.01 billion token corpus packed into fixed 2048 token rows, drawn from Sangraha, Wikipedia, and FineWeb-Edu at a 40% Hindi, 35% English, 25% Marathi mix, encoded with a purpose built 64,000 vocabulary tokenizer trained on the same language mix. Every stage is measured rather than assumed, and every decision is written down in `documentation/`.

## Sections

| # | section | path | what it covers |
|---|---------|------|----------------|
| 1 | Tokenizer | [IndicPrayog_tokenizer/](IndicPrayog_tokenizer/) | 6 tokenizer variants trained and ablated, 5 public baselines compared, final 64k vocabulary tokenizer selected and frozen |
| 2 | Dataset preparation | [dataset_preparation/](dataset_preparation/) | 5 step pipeline from raw HuggingFace downloads to the final packed `uint16` training and validation files |
| 3 | Model | [model/](model/) | architecture implementation, one file at a time, each independently testable |
| 4 | Documentation | [documentation/](documentation/) | full written overview of every phase: why each choice was made, what was measured, what broke |

## Phase status

| phase | scope | state |
|-------|-------|-------|
| 1 | Tokenizer, 64k vocabulary, trained and frozen | done |
| 2 | Dataset, 10.01B tokens packed at 2048 context | done |
| 3 | Model architecture implementation | in progress |
| 4 | Training run | not started |
| 5 | Evaluation | not started |

## Architecture at a glance

| property | value |
|----------|-------|
| total parameters | 502,904,252 |
| active parameters per token | 54,244,796 |
| layers | 19 (1 dense feed forward, 18 Mixture of Experts) |
| hidden size | 768 |
| attention | grouped query attention, 12 query heads, 2 key value heads, head dim 64 |
| positions | RoPE, theta 8,000,000, configured for 131,072 context |
| experts per Mixture of Experts layer | 46 routed plus 1 shared |
| experts active per token | 2 routed plus 1 shared |
| router | sigmoid scoring in fp32, expert bias for load balancing |
| vocabulary | 64,000, untied input embedding and output head |
| training context length | 2048 |

## Dataset at a glance

| property | value |
|----------|-------|
| total training tokens | 10,011,318,272 |
| packed rows | 4,888,339 |
| row length | 2048 tokens |
| storage dtype | `uint16` |
| language mix | Hindi 40%, English 35%, Marathi 25% |
| sources | Sangraha, Wikipedia, FineWeb-Edu |
| validation holdout | 0.5% per language, held out before tokenization |

## License

See [LICENSE](LICENSE).
