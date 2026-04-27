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
  depending on chip generation and text length. The model is accurate but
  compute-heavy — it is best suited for:
    • Offline / batch sanitisation pipelines
    • Low-volume inline use (e.g. inside an AI Agent tool call)
  It is NOT designed for high-volume, low-latency inline data streams.

Architecture
------------
- Single model instance on MPS/GPU — no worker threads hit the model.
  The GPU serialises all batches internally and is far more efficient than
  multiple CPU processes competing for resources.
- A background writer thread consumes completed batches from a queue and
  writes to CSV while the main thread is already running the next inference
  batch — keeping the GPU fed rather than waiting for disk I/O.
- BATCH_SIZE=256 keeps MPS utilisation high (larger batches amortise the
  fixed per-call overhead). Lower to 64 if you hit out-of-memory errors.
- PROGRESS_EVERY and WRITER_QUEUE_DEPTH can be tuned at the top of this file.

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
import queue
import threading
import argparse
from collections import defaultdict
from pathlib import Path

# ── Load .env ─────────────────────────────────────────────────────────────────
# python-dotenv is optional but recommended.  Install with:
#   pip install python-dotenv
# Values already set in the environment always take precedence over .env.
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # fall back to whatever is already in the environment

import pandas as pd
import torch
from transformers import pipeline

# ── Environment-driven config ─────────────────────────────────────────────────
# HF_MODEL_ID     — Hugging Face repo ID (default: openai/privacy-filter)
# LOCAL_MODEL_DIR — path to locally downloaded model weights; when set and the
#                   directory exists, demo.py loads from disk instead of the
#                   global HF cache, making it safe to run behind a proxy.
# TRANSFORMERS_OFFLINE — set to "1" in .env to block all network calls.
#                   The transformers library reads this variable natively.
HF_MODEL_ID     = os.getenv("HF_MODEL_ID",    "openai/privacy-filter")
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR", "")

# Resolve the model source: prefer the local directory if it exists.
_local_path = Path(LOCAL_MODEL_DIR).resolve() if LOCAL_MODEL_DIR else None
MODEL_SOURCE = str(_local_path) if (_local_path and _local_path.is_dir()) else HF_MODEL_ID

# ── Tuneable knobs ────────────────────────────────────────────────────────────
# MPS is Apple Silicon GPU — much faster than CPU for transformer inference.
# Fall back to CPU if unavailable.
if torch.backends.mps.is_available():
    DEVICE = "mps"
elif torch.cuda.is_available():
    DEVICE = "cuda"
else:
    DEVICE = "cpu"

# Larger batches keep MPS fed — 256 is a good starting point for Apple Silicon.
# Lower to 64 if you hit out-of-memory errors.
BATCH_SIZE     = 256
PROGRESS_EVERY = 1000   # print a status line every N rows
WRITER_QUEUE_DEPTH = 8  # max batches buffered between inference and CSV writer
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
    # MODEL_SOURCE is resolved at startup (top of file):
    #   • If LOCAL_MODEL_DIR is set in .env and the directory exists on disk,
    #     the model is loaded directly from that path — no network access at all.
    #   • Otherwise falls back to the HF repo ID, which loads from the global
    #     HF disk cache (~/.cache/huggingface/hub/) if available, or downloads
    #     the weights on first use.
    #
    # Expected model files (in LOCAL_MODEL_DIR or the HF cache snapshot dir):
    #   config.json            — model architecture & label map
    #   tokenizer_config.json  — tokenizer settings
    #   tokenizer.json         — tokenizer vocabulary & rules
    #   model.safetensors      — model weights (~500 MB, bfloat16)
    #
    # To download the model locally before going offline, run:
    #   python download_model.py
    # Then set TRANSFORMERS_OFFLINE=1 and LOCAL_MODEL_DIR in your .env.

    # Warn early if the local model directory is missing so the user sees a
    # clear message rather than a cryptic connection / proxy error.
    offline = os.getenv("TRANSFORMERS_OFFLINE", "0").strip() == "1"
    if offline and MODEL_SOURCE == HF_MODEL_ID:
        # TRANSFORMERS_OFFLINE=1 but no local dir — will likely fail
        print(
            f"⚠  TRANSFORMERS_OFFLINE=1 is set but LOCAL_MODEL_DIR "
            f"('{LOCAL_MODEL_DIR}') was not found on disk.\n"
            "   Run  python download_model.py  first, then retry.\n"
        )
    elif MODEL_SOURCE == HF_MODEL_ID:
        # Using the global HF cache — check it has been populated
        try:
            from huggingface_hub import try_to_load_from_cache, _CACHED_NO_EXIST
            sentinel = try_to_load_from_cache(HF_MODEL_ID, "config.json")
            if sentinel is None or sentinel is _CACHED_NO_EXIST:
                print(
                    f"⚠  Model '{HF_MODEL_ID}' not found in the local HF cache.\n"
                    "   Run  python download_model.py  to download it first.\n"
                )
        except Exception:
            pass  # huggingface_hub not available or cache layout changed — carry on
    else:
        print(f"Model source : {MODEL_SOURCE}  (local)")

    print("Loading model…")
    t0 = time.time()
    clf = pipeline(
        "token-classification",
        model=MODEL_SOURCE,
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

    # ── Background writer thread ──────────────────────────────────────────────
    # The writer thread consumes completed batches from a queue and writes them
    # to CSV while the main thread is already running the next inference batch.
    # This keeps MPS busy instead of waiting for disk I/O.
    write_queue    = queue.Queue(maxsize=WRITER_QUEUE_DEPTH)
    writer_errors  = []

    def csv_writer_thread(fpath):
        try:
            with open(fpath, "w", newline="", encoding="utf-8") as fout:
                writer = csv.writer(fout)
                writer.writerow(["id", "original_text", "sanitized_text"])
                while True:
                    item = write_queue.get()
                    if item is None:          # sentinel — shut down
                        break
                    for row_id, original, sanitized in item:
                        writer.writerow([row_id, original, sanitized])
                    write_queue.task_done()
        except Exception as e:
            writer_errors.append(e)

    writer_thread = threading.Thread(
        target=csv_writer_thread, args=("sanitized_data.csv",), daemon=True
    )
    writer_thread.start()

    # ── Inference loop ────────────────────────────────────────────────────────
    for batch_offset in range(0, total, BATCH_SIZE):
        batch_texts = texts[batch_offset : batch_offset + BATCH_SIZE]
        batch_ids   = ids[batch_offset   : batch_offset + BATCH_SIZE]

        # GPU forward pass — dominates wall time
        batch_results = clf(batch_texts, batch_size=BATCH_SIZE)

        # Build rows and count detections (CPU, fast)
        batch_rows = []
        for i, entities in enumerate(batch_results):
            sanitized = sanitize_text(batch_texts[i], entities)
            batch_rows.append((batch_ids[i], batch_texts[i], sanitized))
            for e in entities:
                key = e["entity_group"].upper()
                detections_by_type[key] += 1
                total_detections        += 1

        # Hand off to writer — non-blocking as long as queue isn't full
        write_queue.put(batch_rows)

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

    # Signal writer to finish and wait for it to flush
    write_queue.put(None)
    writer_thread.join()

    if writer_errors:
        raise RuntimeError(f"CSV writer failed: {writer_errors[0]}")

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
