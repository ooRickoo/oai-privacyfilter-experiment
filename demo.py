#!/usr/bin/env python3
"""
demo.py — OpenAI Privacy Filter Demo  (Apple Silicon / MPS optimised)
=======================================================================
Loads the openai/privacy-filter token-classification model from the local
Hugging Face cache and runs it against the synthetic dataset produced by
generate_data.py.  All inference is performed **locally** — no data leaves
the machine after the one-time model download.

Workflow
--------
1. Auto-detect the best available device (MPS → CUDA → CPU).
2. Load the model and tokenizer from the local HF cache.
3. Read synthetic_data.csv into a pandas DataFrame (sample_raw).
4. Process every row through the privacy filter in explicit mini-batches,
   replacing detected PII spans with typed placeholders, e.g.:
       "Call me at 555-1234"  →  "Call me at [PRIVATE_PHONE]"
5. Stream results row-by-row to sanitized_data.csv (sample_sanitized) so
   memory usage stays flat regardless of dataset size.
6. Print a live progress bar during processing and a full summary on exit.

Detected PII types (OpenAI Privacy Filter label set)
------------------------------------------------------
  PRIVATE_PERSON    PRIVATE_ADDRESS   PRIVATE_EMAIL    PRIVATE_PHONE
  PRIVATE_URL       PRIVATE_DATE      ACCOUNT_NUMBER   SECRET

Performance notes
-----------------
- On Apple Silicon (M-series) with MPS enabled, expect ~10–40 rows/sec
  depending on chip generation and text length.  The model is accurate but
  compute-heavy — it is best suited for:
    • Offline / batch sanitisation pipelines
    • Low-volume inline use (e.g. inside an AI Agent tool call)
  It is NOT designed for high-volume, low-latency inline data streams.
- BATCH_SIZE and PROGRESS_EVERY can be tuned at the top of this file.

Output
------
sanitized_data.csv  — three columns: id, original_text, sanitized_text

Usage
-----
    python demo.py          # uses synthetic_data.csv in the current directory

Dependencies
------------
    pip install pandas torch transformers
"""

import os
import sys
import time
import csv
import argparse
from collections import defaultdict

import pandas as pd
import torch
from transformers import pipeline

# ── Tuneable knobs ────────────────────────────────────────────────────────────
# MPS is Apple Silicon GPU — much faster than CPU for transformer inference.
# Fall back to CPU if unavailable.
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# MPS can handle larger batches than CPU; tune if you hit OOM
BATCH_SIZE     = 64
PROGRESS_EVERY = 500    # print a status line every N rows
# ─────────────────────────────────────────────────────────────────────────────


def sanitize_text(text: str, entities: list) -> str:
    """Replace PII spans with [TYPE] placeholders (right-to-left)."""
    for e in sorted(entities, key=lambda x: x["start"], reverse=True):
        text = text[:e["start"]] + f"[{e['entity_group'].upper()}]" + text[e["end"]:]
    return text


def _bar(pct: float, width: int = 25) -> str:
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


def main():
    parser = argparse.ArgumentParser(description="OpenAI Privacy Filter demo")
    parser.add_argument(
        "--sample", type=int, default=None,
        metavar="N",
        help="Process only the first N rows (omit to process all 300k)"
    )
    args = parser.parse_args()

    print(f"Device : {DEVICE}")
    print(f"Batch  : {BATCH_SIZE}")
    if args.sample:
        print(f"Sample : {args.sample:,} rows (use without --sample for full dataset)")
    print()

    # ── Load model ────────────────────────────────────────────────────────────
    print("Loading model…")
    t0 = time.time()
    clf = pipeline(
        "token-classification",
        model="openai/privacy-filter",
        aggregation_strategy="simple",
        device=DEVICE,
    )
    print(f"Model loaded in {time.time()-t0:.1f}s\n")

    # ── Load raw data ─────────────────────────────────────────────────────────
    print("Loading synthetic_data.csv…")
    df_raw = pd.read_csv("synthetic_data.csv")
    if args.sample:
        df_raw = df_raw.head(args.sample)
    total  = len(df_raw)
    texts  = df_raw["text"].tolist()
    ids    = df_raw["id"].tolist()
    print(f"Loaded {total:,} rows.\n")

    # ── Process + stream to CSV ───────────────────────────────────────────────
    print(f"Processing {total:,} rows → sanitized_data.csv …")
    print(f"{'─'*72}")

    detections_by_type = defaultdict(int)
    total_detections   = 0
    rows_done          = 0
    wall_start         = time.time()

    with open("sanitized_data.csv", "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(["id", "original_text", "sanitized_text"])

        for batch_offset in range(0, total, BATCH_SIZE):
            batch_texts = texts[batch_offset : batch_offset + BATCH_SIZE]
            batch_ids   = ids[batch_offset   : batch_offset + BATCH_SIZE]

            # Run inference on the whole mini-batch at once
            batch_results = clf(batch_texts, batch_size=BATCH_SIZE)

            for i, entities in enumerate(batch_results):
                sanitized = sanitize_text(batch_texts[i], entities)
                writer.writerow([batch_ids[i], batch_texts[i], sanitized])

                for e in entities:
                    key = e["entity_group"].upper()
                    detections_by_type[key] += 1
                    total_detections        += 1

            rows_done += len(batch_texts)

            if rows_done % PROGRESS_EVERY == 0 or rows_done == total:
                elapsed  = time.time() - wall_start
                rps      = rows_done / elapsed if elapsed else 0
                eta      = (total - rows_done) / rps if rps else 0
                pct      = rows_done / total * 100
                print(
                    f"  [{_bar(pct)}] {pct:5.1f}%  "
                    f"{rows_done:>7,}/{total:,}  "
                    f"{rps:>7,.0f} rows/sec  "
                    f"elapsed {elapsed:>5.0f}s  ETA {eta:>5.0f}s  "
                    f"detections {total_detections:,}"
                )

    total_time = time.time() - wall_start
    print(f"{'─'*72}\n")
    print("Exported → sanitized_data.csv")

    # ── Summary ───────────────────────────────────────────────────────────────
    rows_with_pii = sum(1 for t in open("sanitized_data.csv") if "[" in t) - 1  # minus header
    rows_clean    = total - rows_with_pii

    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Device                : {DEVICE}")
    print(f"  Batch size            : {BATCH_SIZE}")
    print(f"  Total rows processed  : {total:,}")
    print(f"  Total wall-clock time : {total_time:.2f}s")
    print(f"  Throughput            : {total / total_time:,.0f} rows/sec")
    print(f"  Avg time per row      : {total_time / total * 1000:.2f} ms")
    print(f"  Rows with PII found   : {rows_with_pii:,}")
    print(f"  Rows clean            : {rows_clean:,}")
    print(f"  Total PII detections  : {total_detections:,}")
    print(f"  Avg detections / row  : {total_detections / total:.2f}")
    print()
    print("  Detections by type:")
    for typ, count in sorted(detections_by_type.items(), key=lambda x: -x[1]):
        print(f"    {typ:<22}: {count:,}")
    print("=" * 60)

    # ── Sample output ─────────────────────────────────────────────────────────
    print("\nSample sanitized rows:")
    df_san = pd.read_csv("sanitized_data.csv", nrows=5)
    for _, row in df_san.iterrows():
        print(f"  Original : {row['original_text']}")
        print(f"  Sanitized: {row['sanitized_text']}")
        print("─" * 60)


if __name__ == "__main__":
    main()
