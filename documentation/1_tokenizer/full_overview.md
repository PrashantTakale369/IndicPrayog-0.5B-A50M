# IndicPrayog Tokenizer - Complete Overview

## What is This?

IndicPrayog uses a **custom 64,000-vocabulary Unigram tokenizer** (SentencePiece-style) built specifically for **Hindi, Marathi, and English**. It was trained from scratch on a balanced Indic + English corpus and validated through rigorous quality gates before being adopted for the IndicPrayog-0.5B model.

This is NOT an off-the-shelf tokenizer (GPT-2, Mistral, Qwen, Gemma): it's purpose-built for morphologically rich Indic languages.

---

## Why We Built a Custom Tokenizer

### The Problem

Hindi and Marathi words are built from **aksharas** (syllables): a base consonant plus vowel marks (*matras*). For example: `भ` + `ा` = `भा`.

General-purpose English tokenizers (GPT-2, Mistral, Qwen, Gemma) don't understand this structure and produce terrible results:

| Issue | Problem |
|-------|---------|
| **Fertility** | 5–8 tokens per Hindi word vs. ~1.2–1.4 for English |
| **Context cost** | Hindi consumes 5–8x more context window for the same meaning |
| **Compute cost** | 5–8x more compute to train/run compared to English |
| **Vocabulary waste** | Most slots wasted on byte-fragments that never form real words |

### Our Solution

Build a tokenizer that is:
- **Compact**: Low tokens-per-word (fertility)
- **Fair**: Low parity gap between Hindi and English
- **Lossless**: 100% round-trip fidelity: `decode(encode(text)) == text`
- **Effective**: Actually improves downstream model learning

---

## Training Corpus

| Language | Size | Share | Documents |
|----------|------|-------|-----------|
| Hindi | 600 MB | 40% | ~250–350K |
| English | 525 MB | 35% | ~120–180K |
| Marathi | 375 MB | 25% | ~180–250K |
| **Total** | **1.5 GB** | **100%** | **~550–780K** |

### Data Sources

1. **Sangraha (Verified)**: Highest-quality open Indic corpus from IndicLLMSuite (AI4Bharat)
2. **FineWeb-Edu**: Best open English data; CommonCrawl filtered by educational quality classifier

**Key decision**: Byte-budget sampling ensures balanced representation. Devanagari is 3 UTF-8 bytes per character, so sampling by document count would silently give English 3x more information. We sample by byte to balance fairly.

**Language mix note**: Marathi upweighted to 25% (vs. model's 20%) to protect Marathi subword inventory, since BPE/Unigram merges are frequency-driven. At tokenizer-training time, this prevents Hindi-specific subwords from crowding out Marathi ones.

---

## How We Built It: The 6-Way Ablation

To isolate the impact of each design choice, we trained **6 variants on identical data**, changing exactly one factor at a time:

| Variant | What Changed | Result |
|---------|-------------|--------|
| **A** | BPE, mix 40/35/25, ByteLevel + digit-split | Balanced baseline |
| B | BPE, equal mix 33/33/34 | Rejected (integrity dropped) |
| C | BPE, raw mix 45/35/20 | Shortlisted for comparison |
| **D** | **Unigram (SentencePiece)** ✓ WINNER | **Best performance** |
| E | BPE + UnicodeScripts boundary splitting | Rejected (broke integrity) |
| F | BPE, no digit-splitting | Rejected (integrity threshold missed) |

### The Shortlisting Gate

Before expensive proxy-LM comparison, a pre-registered rule reduced 6 to 3:
- **Top 3 by mean fertility** (tokens per word)
- **Subject to**: Round-trip fidelity = 100% (non-negotiable: lossy tokenizers disqualified)
- **Subject to**: Devanagari integrity ≥ Variant A baseline (93.2%)

**Winners: A, C, D** proceeded to proxy-LM testing.

---

## Three Critical Bugs We Found (and Fixed)

These silently degraded Devanagari quality: most Indic tokenizers likely carry the same bugs:

### Bug 1: `add_prefix_space=True` injects spurious spaces
The library default inserts a space marker at **every** pre-tokenizer boundary, not just string start.
- **Example**: `"पास4महीने"` became `['Ġपास', 'Ġ4', 'Ġमहीने']`: fabricated space before digit
- **Impact**: Broke round-trip fidelity and inflated fertility
- **Fix**: `ByteLevel(add_prefix_space=False)`

### Bug 2: `UnicodeScripts()` silently drops whitespace
Running `UnicodeScripts()` after `Digits()` means a space between a digit and next word gets deleted.
- **Example**: `"पास 4 महीने"` became `"पास4महीने"`
- **Fix**: Run `UnicodeScripts()` **before** `Digits()`

### Bug 3: ByteLevel's regex splits Devanagari at every matra (the big one)
ByteLevel groups Letter (L) category characters; Devanagari marks (matras) are Mark (Mc/Mn), so they split mid-akshara.
- **Example**: `भारत` (1 cluster) pre-split to `['भ', 'ा', 'रत']`: BPE could never merge `भ+ा`
- **Impact**: Hindi/Marathi fertility 3.3–5 tokens/word; Devanagari integrity only 40%
- **Fix**: Custom regex `r" ?[^\s\d]+| ?\d+|\s+"` groups Letter+Mark together; use `ByteLevel(use_regex=False)`

### Bug 4: Model initialization (found during proxy-LM)
PyTorch's default embedding initialization (`N(0,1)`) caused 28x too-high step-0 loss.
- **Fix**: GPT-2/Llama-style init `N(0, 0.02)` for Linear and Embedding weights

---

## Tokenizer Architecture & Configuration

### Base Config (All Variants)

| Setting | Value | Why |
|---------|-------|-----|
| Algorithm | Byte-level BPE (or Unigram for Variant D) | Fast, works well with byte fallback |
| Vocabulary | 64,000 | Matches model's `vocab_size` |
| Normalizer | NFC | Canonicalizes Devanagari matra encodings |
| Pre-tokenizer | `Digits(individual_digits=True)` + custom Letter+Mark regex | Splits `2024→2 0 2 4`; keeps matras intact |
| Decoder | ByteLevel | Inverse of pre-tokenizer |
| `min_frequency` | 2 | Merge must occur ≥2x in corpus |
| `initial_alphabet` | `ByteLevel.alphabet()` | All 256 raw bytes guaranteed |
| Special tokens | `<pad>`, `<bos>`, `<eos>`, `<unk>` (ids 0–3) | Standard pipeline requirements |
| `model_max_length` | 131,072 | Matches model's `max_position_embeddings` |

### Variant D (Winner): Unigram Instead of BPE

- **BPE**: Greedy bottom-up: repeatedly merge the most frequent adjacent pair
- **Unigram**: Probabilistic top-down: start from large vocab, remove tokens whose removal increases loss least: optimize global unigram language-model loss

**Why Unigram wins for morphological richness:**
- Keeps linguistically meaningful subwords intact (e.g., conjunct `प्र`) more consistently
- Survival depends on frequency across all corpus positions, not just one bigram
- Metaspace pre-tokenizer (uses `▁` for whitespace) handles leading spaces natively

---

## Evaluation: 6 Intrinsic Metrics

### 1. **Fertility** (tokens per word): lower is better
How many tokens to encode one word on average. Lower = longer effective context, cheaper compute.

### 2. **Parity Score**: closer to 1.0 is fairer
For parallel sentences (identical meaning), how many more tokens does Hindi/Marathi need vs. English?
- Parity 4.5 = that language pays 4.5x more tokens for the same thought
- Our Variant D: 1.21 (Hindi), 1.09 (Marathi): essentially fair

### 3. **Round-Trip Fidelity**: must be 100%
`decode(encode(text)) == text` for every sentence. Anything < 100% = lossy tokenizer (disqualifying).

### 4. **Devanagari Integrity**: higher is better
% of Devanagari grapheme clusters (aksharas, including conjuncts) that survive as a contiguous span inside a single token.

### 5. **Vocabulary Utilization**: higher is better
What fraction of 64,000 slots actually used in real text? Low utilization = wasted slots.

### 6. **Compression** (bytes per token): higher is better
Average raw bytes each token represents.

### Proxy-LM Evaluation (Top 3 Candidates Only)
Intrinsic metrics measure compression, not whether low-fertility actually helps models learn. For that, we **trained a real small language model** on each shortlisted tokenizer and compared using **bits-per-byte (BPB)**:
- BPB normalizes by raw UTF-8 bytes (not tokens) so all three models fairly compared on reconstructing the same underlying bytes
- Model: 28.1M-parameter Llama-style, 3,000 steps, 32K tokens/step

---

## Results: Why Variant D Wins

### Raw Metrics (FLORES-200 devtest)

```
Tokenizer         | Hi Fert | Mr Fert | En Fert | Hi Parity | Mr Parity | Integrity | RT
------------------+---------+---------+---------+-----------+-----------+-----------+-----
ours_D (WINNER)   |  1.27   |  1.54   |  1.23   |   1.21    |   1.09    |  96.0%    | 100%
ours_A            |  1.24   |  1.54   |  1.24   |   1.17    |   1.08    |  93.2%    | 100%
ours_C            |  1.24   |  1.56   |  1.24   |   1.17    |   1.09    |  93.3%    | 100%
sarvam-1          |  1.40   |  1.77   |  1.43   |   1.14    |   1.07    |  95.6%    | 100%
gpt2              |  7.82   | 11.16   |  1.23   |   7.41    |   7.86    |  39.9%    | 100%
```

### Against General-Purpose Baselines

**vs. GPT-2**: 6.2x better on Hindi (1.27 vs. 7.82 tokens/word)
**vs. Mistral**: 4.2x better on Hindi (1.27 vs. 5.33)
**vs. Gemma-2**: 1.5x better on Hindi; 2.1x better on Marathi

### Against Sarvam-1 (Indic Gold Standard, 68K vocab)

- **Our D wins on fertility** across all three languages with a *smaller* vocabulary
- Hindi: 1.27 vs. 1.40 (−9.3%)
- Marathi: 1.54 vs. 1.77 (−13%)
- English: 1.23 vs. 1.43 (−14%)
- **Devanagari integrity**: 96.0% vs. 95.6%

The one metric Sarvam-1 edges ahead: Hindi parity (1.14 vs. our 1.21): marginally fairer. But on raw compression (saves context and compute), Variant D wins across the board.

### Proxy-LM Results (BPB: Lower is Better)

| Variant | Mean BPB | Hindi | English | Marathi |
|---------|----------|-------|---------|---------|
| **D (winner)** | **1.0404** | **0.7229** | **1.6664** | **0.7320** |
| A | 1.0671 | 0.7324 | 1.7201 | 0.7489 |
| C | 1.0686 | 0.7316 | 1.7226 | 0.7516 |

Variant D wins **on every language individually**: not a tradeoff. 2.5% advantage over A, 2.7% over C. This confirms intrinsic ranking; actual models trained with D learn better than with A or C.

---

## Training Pipeline (6 Steps)

### Step 1: Sample Corpus (Streaming)
- Stream each language from HuggingFace with `streaming=True` (no full download)
- Stop when byte target hit (600 MB Hindi, 525 MB English, 375 MB Marathi)
- Clean (collapse newlines), skip docs < 150 chars
- Resume-friendly: skip language if already ≥98% of target

### Step 2: Train Tokenizer
- Train all three language files together (so BPE/Unigram merges compete across languages)
- Apply all bug fixes: `add_prefix_space=False`, custom regex, NFC norm
- Target 64,000 vocab, `min_frequency=2`
- Takes 15–40 minutes on CPU

### Step 3: Validate Fertility (Pass/Reject Gates)
Before proceeding, test on held-out text:
- Hindi target ~1.8, hard reject > 2.5
- Marathi target ~2.0, hard reject > 2.8
- English target ~1.3, hard reject > 1.8
- Round-trip fidelity check

### Step 4: Save in HuggingFace Format
Wrap `tokenizer.json` in `transformers.PreTrainedTokenizerFast` with special tokens set, save with `save_pretrained()`.

### Step 5: Inspect (Playground)
Run arbitrary text through tokenizer, print token pieces: sanity-check that segmentation looks linguistically reasonable.

### Step 6: Publish & Clean Up
Upload to HuggingFace, delete ~1.5 GB sampled corpus: only < 5 MB trained tokenizer needed.

---

## Quick Start

### Load Pre-Trained Tokenizer

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("path/to/tokenizer")

text = "भारत एक विशाल देश है।"
ids = tok(text).input_ids
print(ids)
print(tok.decode(ids))
```

### Reproduce Training from Scratch

```bash
pip install -q datasets tokenizers transformers tqdm
```

Then run the notebook (`IndicPrayog-0.5B Custom 64K Tokenizer Pipeline`) top to bottom.

### Reproduce Full Experiment (6-Way Ablation + Proxy-LM)

1. Repeat corpus sampling + training for each variant (A–F)
2. Apply shortlisting gate: 3 finalists
3. Train 28M-parameter proxy-LM on each using identical data/seed/hyperparams
4. Select lowest mean BPB candidate

---

## Why We Trust These Results

1. ✓ Standard public benchmark (FLORES-200, used community-wide)
2. ✓ Same code, same data for every tokenizer
3. ✓ No eval contamination (FLORES never in training corpus)
4. ✓ Identical proxy-LM conditions (architecture, hyperparams, seed, docs)
5. ✓ Step-0 sanity check (every run verified to start near `ln(64000) = 11.07`)
6. ✓ Pre-registered selection rule (fixed before seeing results)
7. ✓ All bugs disclosed (three tokenizer-config bugs + init bug documented)

---

## Key Takeaways

- **Unigram beats BPE** on morphologically rich languages: global optimization wins over greedy merging
- **Three silent bugs** nearly destroyed Devanagari quality; fixed at the config level
- **Variant D** outperforms Sarvam-1 (Indic gold standard) on fertility with smaller vocab
- **Proxy-LM validation** confirmed intrinsic metrics predict real model learning
- **The tokenizer is permanent**: once frozen for pretraining, changing it means retraining from scratch; choosing right now was worth the ablation cost

---

## Directory Structure

```
IndicPrayog_tokenizer/
├── dataset/
│   └── tokenizer_dataset/        (deleted after training)
│       ├── hi.txt    (600 MB)
│       ├── en.txt    (525 MB)
│       └── mr.txt    (375 MB)
└── tokenizer/                    (kept forever: < 5 MB)
    ├── tokenizer.json
    ├── tokenizer_config.json
    └── special_tokens_map.json
```

---

## Publication & Ownership

- **Published on**: HuggingFace (privately) as `prashantcp8/IndicPrayog-tokenizer-64k`
- **Transfer pending**: To AxisQuant organization (awaiting write access)

---

## Key Takeaway for This Overview

This document consolidates the full tokenizer story into one file: problem, data, ablation, bugs found and fixed, architecture, evaluation methodology, results, and how to use it. It is the single reference for understanding how the IndicPrayog tokenizer was built and why it was chosen.
