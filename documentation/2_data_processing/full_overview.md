# IndicPrayog Dataset Pipeline - Complete Overview

## What is This?

The dataset pipeline turns raw, differently-shaped HuggingFace downloads (Sangraha, Wikipedia, FineWeb-Edu) into the final **10 billion token**, `uint16`, fixed length training and validation files the IndicPrayog-0.5B model trains on directly. Five steps, each with its own notebook/script under `dataset_preparation/dataset_complete_pipeline/`, each writing its own output folder so every stage is inspectable on its own.

---

## Why This Pipeline Exists

Three open datasets, three different shapes: different column names, different or missing `title`/`url` fields, different document-type tags. Feeding that straight into tokenization would mean bespoke logic at every later step. Beyond schema, two real risks needed measuring, not assuming:

- **Language purity**: Hindi and Marathi both use Devanagari script, so a folder name (`sangraha/verified/hin/`) is not proof of what language a row actually contains.
- **Token budget**: the model needs a fixed training token count, and Hindi/Marathi cost more tokens per character than English with this tokenizer, so the real total had to be measured, not estimated.

---

## Data Sources

| Source | Domain | Languages | Notes |
|--------|--------|-----------|-------|
| Sangraha (verified) | web | hi, mr | Largest source by far, web-crawled |
| Wikipedia | encyclopedia | hi, mr, en | Small, curated, kept in full at every step |
| FineWeb-Edu | web (education-filtered) | en | English's bulk source |

---

## Canonical Schema (Step 1 output, every source normalized to this)

```
id, text, language, source, domain, url, title, metadata
```

| field | meaning |
|-------|---------|
| id | unique document id, prefixed by source and language |
| text | the raw document text |
| language | `hi`, `mr`, or `en` |
| source | `sangraha`, `wikipedia`, or `fineweb-edu` |
| domain | `web`, `pdf`, `speech`, or `encyclopedia` |
| url | source URL if available, else null |
| title | document title if available, else null |
| metadata | JSON string of anything source specific |

---

## The 5 Steps

| step | folder | does | status |
|------|--------|------|--------|
| 1 | `1_format_standardize/` | raw sources → canonical schema | done |
| 2 | `2_lang_validation/` | tag every row with a real, model-checked language | done |
| 3 | `3_train_and_val_split/` | cut 18.41B measured tokens down to a 10B budget at a fixed language mix | done |
| 4 | `4_tokenization/` | measure tokenizer efficiency + get the real, exact token count | done |
| 5 | `5_final_dataset/` | tokenize, hold out validation, pack into fixed 2048-token training rows | done |

### Step 1: Format standardization

Rewrites every source into the canonical schema above so every later step works against one predictable shape. Reads from the sibling `dataset/` folder, writes to `dataset/standardized/<language>/<source>/`.

### Step 2: Language validation

Runs fastText `lid.176` on every row (first 1000 characters, enough signal for language ID) and adds `predicted_language` + `language_confidence` columns, without dropping anything. A sanity check found mismatch rates under 1% and mean confidence above 0.94 for every language/source, so this was a minor, well behaved risk, not a real problem. Output: `dataset/lang_validated/<language>/<source>/`, ~25-30 minutes.

### Step 3: Subsample to a 10B token budget

Real full-dataset measurement (Step 4) found **18.41 billion tokens** total, far more than the 10B token budget the model needs. Target language mix: **Hindi 40% / English 35% / Marathi 25%** (matches the tokenizer's own training mix, so there is only one language ratio decision across the whole project, not two slightly different ones).

**How the drop is decided:** for each language, the small curated source (`wikipedia`) is always kept 100%, only the large source (`sangraha`/`fineweb-edu`) is trimmed to whatever fits in the remaining budget. No Wikipedia row is ever dropped.

| language | source | keep_ratio | rows kept | rows dropped | tokens kept (est.) |
|----------|--------|-----------|-----------|---------------|---------------------|
| hi | sangraha | 0.408 | 7,102,415 | 10,318,517 | 3.93B |
| hi | wikipedia | 1.000 | 163,093 | 0 | 0.07B |
| en | fineweb-edu | 0.543 | 2,788,635 | 2,342,365 | 2.80B |
| en | wikipedia | 1.000 | 781,445 | 0 | 0.70B |
| mr | sangraha | 0.877 | 5,144,381 | 721,236 | 2.48B |
| mr | wikipedia | 1.000 | 94,133 | 0 | 0.02B |
| **Total** | | | **16,074,090** | **13,382,130** | **~10.00B** |

Output: `dataset/sampled_10b/<language>/<source>/`, ~6 minutes.

### Step 4: Tokenizer evaluation

Two questions: how efficiently does the tokenizer (`AxisQuant/IndicPrayog-tokenizer-64k`, 64,000 vocab) encode each language, and what is the exact total token count of the full dataset.

**Sample-based efficiency (200K characters per language):**

| category | chars_per_token | tokens_per_word |
|----------|------------------|------------------|
| numbers | 2.08 | 3.79 |
| code | 2.13 | 4.81 |
| punctuation | 2.89 | 2.03 |
| urls | 2.94 | 17.60 |
| names | 3.60 | 2.05 |
| mixed_language | 3.80 | 1.72 |
| hindi | 3.99 | 1.29 |
| marathi | 4.44 | 1.55 |
| english | 4.78 | 1.29 |

**Full dataset token count (real run, all 149 standardized shards, not a sample):**

| language | rows | characters | tokens | chars_per_token |
|----------|------|------------|--------|------------------|
| en | 5.91 million | 27.05 billion | 5.85 billion | 4.62 |
| hi | 17.58 million | 38.51 billion | 9.71 billion | 3.96 |
| mr | 5.96 million | 12.82 billion | 2.85 billion | 4.50 |
| **Total** | **29.46 million** | **78.37 billion** | **18.41 billion** | |

Hindi is more than half of all rows but not half of all tokens: it compresses into fewer characters per token than English, but each document is shorter on average, so the two effects roughly cancel out. Runs on CPU only, ~2 hours.

### Step 5: Tokenize and pack the final dataset

1. Holds out **0.5%** of rows per language as validation, before any tokenization, so the model never sees them.
2. Tokenizes everything with the frozen tokenizer.
3. Packs token ids into fixed **2048 token** rows (model context length), with `<eos>` (id 1) separating documents.
4. Writes flat binary `uint16` files: `train.bin` (all languages, shard order shuffled so batches aren't one language at a time) and one `val_<language>.bin` per language.

**Result of the real run:**

| file | rows | tokens | tail tokens dropped |
|------|------|--------|-----------------------|
| train.bin | 4,863,624 | 9,960,701,952 | 597 |
| val_en.bin | 8,601 | 17,614,848 | 1,189 |
| val_hi.bin | 9,890 | 20,254,720 | 1,672 |
| val_mr.bin | 6,224 | 12,746,752 | 1,244 |
| **Total** | **4,888,339** | **10,011,318,272** | **4,702** |

~10.01B tokens, within 0.1% of the 10B target. Tail tokens dropped are just the last incomplete buffer per file (always under 2048), negligible. Output: `dataset/final_packed/`, ~65 minutes on CPU.

`final_lookup.ipynb` (same folder) decodes random rows from each `.bin` file back to text, splitting on `<eos>` to show the individual documents packed into a row: the actual sanity check that what is on disk is what it claims to be.

---

## Deviations From the Original Plan (Recorded, Not Silently Changed)

The original master plan (`row_not_push_material/final_plan.md`) targeted Hindi 45% / English 35% / Marathi 20%, plus quality filtering and MinHash/LSH dedup as required steps. Actual decisions made and documented in the plan doc:

- **Language mix kept at 40/35/25**, matching the tokenizer's own training mix, instead of switching back to 45/35/20.
- **Quality filtering and deduplication skipped** for this run, to move faster to a trained checkpoint. Known, accepted gap: Sangraha (web-crawled) may contain boilerplate or near-duplicate documents. Revisit if training curves or eval results look off (suspiciously fast loss drops, memorization-like generations).

---

## Final Numbers, End to End

| stage | rows | tokens |
|-------|------|--------|
| Full standardized dataset (Step 1/4) | 29.46 million | 18.41 billion |
| Subsampled to budget (Step 3) | 16.07 million | ~10.00 billion (est.) |
| Final packed train + val (Step 5) | 4.89 million packed rows | 10.01 billion (exact) |

---

## Directory Structure (current, on disk)

```
dataset/
├── sampled_10b/          34G   Step 3 output, kept: only surviving copy of the sampled text
│   └── <language>/<source>/*.parquet
├── final_packed/         19G   Step 5 output: the actual model inputs
│   ├── train.bin
│   ├── val_en.bin
│   ├── val_hi.bin
│   ├── val_mr.bin
│   └── final_pack_report.csv
└── .models/              126M  fastText lid.176 model (Step 2)
```

Raw downloads (`sangraha`, `fineweb-edu`, `wikipedia`) and the Step 1/2 intermediate outputs (`standardized`, `lang_validated`) were deleted after Step 3 to free ~190GB of disk. Consequence: any future change to sampling ratio, tokenizer, or a Step 1/2 bug fix requires re-downloading from HuggingFace and redoing Steps 1-2 from scratch.

---

## How to Reproduce

```bash
cd dataset_preparation/dataset_complete_pipeline/1_format_standardize && python standardize_format.py
cd ../2_lang_validation && jupyter nbconvert --to notebook --execute --inplace lang_validation.ipynb --ExecutePreprocessor.timeout=-1
cd ../3_train_and_val_split && jupyter nbconvert --to notebook --execute --inplace subsample_to_budget.ipynb --ExecutePreprocessor.timeout=-1
cd ../4_tokenization && jupyter nbconvert --to notebook --execute --inplace tokenizer_eval.ipynb --ExecutePreprocessor.timeout=-1
cd ../5_final_dataset && jupyter nbconvert --to notebook --execute --inplace pack_dataset.ipynb --ExecutePreprocessor.timeout=-1
```

Step 4 needs to run before Step 3 the first time (Step 3 reads Step 4's per-shard token counts to compute keep ratios).

---

## Key Takeaways

- One canonical schema at Step 1 means every later step is source-agnostic.
- Language folder names were not trusted; Step 2 measured actual language with a real model instead of assuming.
- The 10B token budget and its 40/35/25 mix are exact, measured decisions, not proportional shrinkage or guesses: the keep ratio per source was solved algorithmically to hit the target.
- The final packed dataset lands at 10.01B tokens, within 0.1% of target, with per-language validation held out before any tokenization ever ran.
- Quality filtering and dedup are a known, intentional gap for this run, not an oversight, documented in the master plan for future revisit.
