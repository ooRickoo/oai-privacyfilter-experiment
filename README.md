# OpenAI Privacy Filter — Local Demo

A Python experiment that runs the [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) model locally to detect and redact personally identifiable information (PII) from free-form text.

---

## About the Model

### OpenAI Privacy Filter

> 📄 [Introducing OpenAI Privacy Filter](https://openai.com/index/introducing-openai-privacy-filter/) — OpenAI Blog  
> 🤗 [openai/privacy-filter](https://huggingface.co/openai/privacy-filter) — Hugging Face Model Card  
> 📋 [Official Model Card (PDF)](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf)  
> 🧪 [Live Demo (Hugging Face Space)](https://huggingface.co/spaces/openai/privacy-filter)  
> 💻 [Source Code](https://github.com/openai/privacy-filter)

OpenAI Privacy Filter is a **bidirectional token-classification transformer** purpose-built for PII detection and masking. It was designed for high-throughput data sanitisation workflows where teams need a model they can run **on-premises** that is fast, context-aware, and tunable.

#### How it works

The model is pretrained autoregressively (similar architecture to `gpt-oss`) then converted into a bidirectional token classifier and post-trained with a supervised classification loss:

- Rather than generating text token-by-token, it **labels an entire input sequence in a single forward pass**.
- A **constrained Viterbi decoder** then decodes coherent spans using BIOES boundary tags (`B-`, `I-`, `E-`, `S-` = Begin, Inside, End, Single).
- The output head emits **33 logits per token** — 1 background class `O` plus 4 boundary tags × 8 PII categories.

#### Key model highlights

| Property | Detail |
| :--- | :--- |
| Architecture | Pre-norm transformer encoder stack, 8 transformer blocks |
| Attention | Grouped-query attention (14 query heads, 2 KV heads), rotary positional embeddings, banded window of 257 tokens |
| Feed-forward | Sparse mixture-of-experts, 128 experts total, top-4 routing per token |
| Parameters | 1.5B total, ~50M active per forward pass |
| Context window | 128,000 tokens (no chunking needed for long documents) |
| License | Apache 2.0 |
| Languages | Primarily English; some multilingual robustness |

Because only ~50M parameters are active per token thanks to MoE routing, inference is significantly faster than the raw 1.5B parameter count suggests.

---

## What We Built

| File | Purpose |
| :--- | :--- |
| `generate_data.py` | Generates 300,000 rows of synthetic, clearly-labelled text covering all 8 PII types |
| `demo.py` | Loads the model, processes the dataset in batches, streams sanitised output to CSV, and prints a live progress dashboard + summary |
| `synthetic_data.csv` | Input — raw synthetic rows (safe to commit; all data is fake) |
| `sanitized_data.csv` | Output — original text alongside the redacted version |

---

## PII Types Detected

The model identifies 8 categories of sensitive data:

| Label | Example |
| :--- | :--- |
| `PRIVATE_PERSON` | Alice Johnson |
| `PRIVATE_ADDRESS` | 123 Main St, Springfield, IL |
| `PRIVATE_EMAIL` | `alice@example.com` |
| `PRIVATE_PHONE` | 555-867-5309 |
| `PRIVATE_URL` | `https://alice.mysite.com` |
| `PRIVATE_DATE` | January 1, 1990 |
| `ACCOUNT_NUMBER` | GB29NWBK6016 |
| `SECRET` | p@ssw0rd! |

---

## Example Output

**Input:**

```text
[SYNTHETIC DATA] Hi Alice Johnson, your order from January 15, 2023 is ready.
Tracking at https://track.mysite.com, account GB29NWBK6016.
Contact: alice@example.com or 555-867-5309.
```

**Output:**

```text
[SYNTHETIC DATA] Hi [PRIVATE_PERSON], your order from [PRIVATE_DATE] is ready.
Tracking at [PRIVATE_URL], account [ACCOUNT_NUMBER].
Contact: [PRIVATE_EMAIL] or [PRIVATE_PHONE].
```

---

## Setup

```bash
# 1. Create and activate a virtual environment
python3 -m venv venv
source venv/bin/activate

# 2. Install dependencies
pip install pandas faker torch transformers

# 3. Generate synthetic data (one-time, ~1 minute)
python generate_data.py

# 4. Run the privacy filter demo
python demo.py
```

> **First run:** the model weights (~2.8 GB) are downloaded from Hugging Face and cached locally at  
> `~/.cache/huggingface/hub/models--openai--privacy-filter/`.  
> All subsequent runs are **fully offline** — no data leaves the machine after the initial download.

---

## Script Documentation

### `generate_data.py`

Generates a realistic-looking synthetic dataset using the [Faker](https://faker.readthedocs.io/) library and pre-defined sentence templates.

- **300,000 rows** total
- **70%** of rows contain one or more PII fields drawn from all 8 supported categories
- **30%** of rows contain only benign, non-PII text — giving the model something it should correctly leave untouched
- Every row is prefixed with `[SYNTHETIC DATA]` so it is unambiguous that no real personal data is present
- Output: `synthetic_data.csv` with columns `id`, `text`

```bash
python generate_data.py
```

---

### `demo.py`

Loads the `openai/privacy-filter` model and processes the synthetic dataset end-to-end.

**Workflow:**

1. Auto-detects the best available compute device: **MPS** (Apple Silicon GPU) → **CUDA** → **CPU**
2. Loads the model and tokenizer from the local Hugging Face cache
3. Reads `synthetic_data.csv` into a `sample_raw` pandas DataFrame
4. Processes every row through the privacy filter in explicit mini-batches (`BATCH_SIZE = 64`)
5. Replaces detected PII spans with typed placeholders using right-to-left offset replacement to keep character positions valid:

   ```text
   "Call me at 555-1234"  →  "Call me at [PRIVATE_PHONE]"
   ```

6. Streams results row-by-row to `sanitized_data.csv` (`sample_sanitized`) — memory stays flat regardless of dataset size
7. Prints a live ASCII progress bar with rows/sec and ETA during processing
8. Prints a full summary on completion: throughput, detection counts by type, and sample before/after rows

```bash
python demo.py
```

**Tuneable constants at the top of the file:**

| Constant | Default | Purpose |
| :--- | :--- | :--- |
| `BATCH_SIZE` | `64` | Rows per model call — increase to `128`/`256` on MPS if memory allows |
| `PROGRESS_EVERY` | `500` | How often (in rows) to print a progress line |

---

## Performance

Testing was conducted on Apple Silicon (M-series) using the MPS backend.

| Metric | Observed |
| :--- | :--- |
| Device | Apple MPS (GPU) |
| Batch size | 64 |
| Dataset size | 300,000 rows |
| Throughput | ~10–40 rows/sec |
| Avg time per row | ~25–100 ms |
| PII detection rate | ~70% of rows (matches synthetic data composition) |

The demo auto-selects the best available device at runtime.

---

## Key Findings

### ✅ Strengths

- **Significantly more accurate than regex** — The model understands *context*, not just patterns. It correctly identifies names, dates, and addresses that regex would miss entirely (e.g. "Alice" in context vs. a company name), and avoids false positives on things like product codes that look like phone numbers.
- **8 semantic PII categories in one model** — A single forward pass covers what would otherwise require 8+ carefully maintained and brittle regular expressions, each with their own edge cases and regional variations.
- **Runs 100% locally after download** — Zero data leaves the machine at inference time. Ideal for sensitive or air-gapped environments.
- **No API key, no cost per call** — Unlike cloud-based PII APIs (AWS Comprehend, Google DLP, etc.), there is no per-request pricing and no external availability dependency.
- **Long context window (128k tokens)** — Long documents can be processed in a single call without chunking.
- **Apache 2.0 licence** — Free for commercial use, fine-tuning, and redistribution.

### 🤖 Ideal fit: AI Agent integration

The model's single-pass labelling design makes it a natural **tool inside an AI Agent pipeline**:

- Load once at agent startup; subsequent calls are just a forward pass (~25–100 ms per message)
- Call it as a tool before storing user input, before logging LLM output, or before forwarding data to a third-party service
- No network hop required — keeps sensitive data local throughout the entire agent workflow

### ⚠️ Considerations & Limitations

- **Not designed for high-volume inline streams** — At ~10–40 rows/sec on Apple Silicon, processing a 300k-row dataset takes hours. This makes it unsuitable as an inline filter on a high-throughput message bus or real-time API gateway.
- **Model size** — 2.8 GB on disk; requires meaningful RAM and a ~5–15 second warm-up on first load.
- **CPU is slow** — Without MPS or CUDA, expect 2–5 rows/sec. A GPU is strongly recommended for batch workloads.
- **Not a compliance guarantee** — As OpenAI's own model card states: *"Privacy Filter is a redaction and data minimization aid, not an anonymization, compliance, or safety guarantee."* It should be one layer in a broader privacy-by-design approach.
- **Static label policy** — The model's 8 categories and decision boundaries are fixed at training time. Organisation-specific policies (e.g. redacting job titles or internal project codes) require fine-tuning.
- **English-first** — Performance may drop on non-English text, non-Latin scripts, or regional naming conventions underrepresented in training data.
- **Known failure modes** (from the [official model card](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf)):
  - Under-detection of uncommon names, initials, or domain-specific identifiers
  - Over-redaction of public entities or organisations when local context is ambiguous
  - Fragmented span boundaries in mixed-format or heavily punctuated text
  - Missed secrets for novel credential formats or tokens split across syntax

### 📊 Ideal Use Cases

| Use Case | Fit |
| :--- | :--- |
| AI Agent tool — sanitise user input before storing | ✅ Excellent |
| AI Agent tool — redact LLM output before forwarding | ✅ Excellent |
| Offline batch sanitisation of historical data | ✅ Good (overnight job) |
| Document review pipeline (legal, HR, medical) | ✅ Good with human review |
| Real-time API gateway PII filter | ❌ Too slow |
| High-volume Kafka / message stream processing | ❌ Too slow |

---

## Architecture

```
generate_data.py
  └── Faker + templates → synthetic_data.csv (300k rows, 8 PII types, 70/30 split)

demo.py
  ├── Auto-detect device (MPS / CUDA / CPU)
  ├── Load openai/privacy-filter (local HF cache, ~2.8 GB)
  ├── Read synthetic_data.csv  →  sample_raw DataFrame
  ├── Batch inference loop (batch_size=64)
  │     ├── clf(batch_texts)  →  list of entity spans per row
  │     └── sanitize_text()   →  right-to-left span replacement → [LABEL]
  ├── Stream write row-by-row → sanitized_data.csv (sample_sanitized)
  └── Summary: throughput · detection counts by type · sample rows
```

---

## Dependencies

```
pandas
faker
torch
transformers
```

Install all with:

```bash
pip install pandas faker torch transformers
```

---

## Further Reading

- [Introducing OpenAI Privacy Filter](https://openai.com/index/introducing-openai-privacy-filter/) — OpenAI Blog announcement
- [openai/privacy-filter on Hugging Face](https://huggingface.co/openai/privacy-filter) — Model card, architecture details, usage examples
- [Official Model Card (PDF)](https://cdn.openai.com/pdf/c66281ed-b638-456a-8ce1-97e9f5264a90/OpenAI-Privacy-Filter-Model-Card.pdf) — Bias, risks, limitations, and evaluation methodology
- [Live Demo](https://huggingface.co/spaces/openai/privacy-filter) — Try it in the browser without installing anything
- [Source Code](https://github.com/openai/privacy-filter) — OpenAI's official GitHub repository
