#!/usr/bin/env python3
"""
download_model.py — Download the openai/privacy-filter model from Hugging Face
===============================================================================
Run this ONCE while you still have internet access (or before going behind a
proxy that blocks huggingface.co).  The model weights and tokenizer files are
saved to the LOCAL_MODEL_DIR path configured in your .env file (default:
./model/openai-privacy-filter).

After this script completes, set TRANSFORMERS_OFFLINE=1 in your .env and all
subsequent runs of demo.py will load entirely from local disk — no network
requests are made.

Usage
-----
    python download_model.py

Optional environment variables (set in .env or export before running):
    HF_MODEL_ID       — Hugging Face model repo ID  (default: openai/privacy-filter)
    LOCAL_MODEL_DIR   — where to save model files   (default: ./model/openai-privacy-filter)
    HF_TOKEN          — Hugging Face access token, if the repo requires auth
"""

import os
import sys

# ── Load .env if present ──────────────────────────────────────────────────────
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed — fall back to plain env vars

from pathlib import Path

HF_MODEL_ID     = os.getenv("HF_MODEL_ID",     "openai/privacy-filter")
LOCAL_MODEL_DIR = os.getenv("LOCAL_MODEL_DIR",  "model/openai-privacy-filter")
HF_TOKEN        = os.getenv("HF_TOKEN") or None   # None → unauthenticated

dest = Path(LOCAL_MODEL_DIR).resolve()
dest.mkdir(parents=True, exist_ok=True)

print(f"Model ID  : {HF_MODEL_ID}")
print(f"Saving to : {dest}")
if HF_TOKEN:
    print("Auth      : HF_TOKEN set ✓")
print()

try:
    from transformers import AutoTokenizer, AutoModelForTokenClassification
except ImportError:
    sys.exit("transformers is not installed.  Run:  pip install -r requirements.txt")

try:
    from huggingface_hub import snapshot_download
except ImportError:
    sys.exit("huggingface_hub is not installed.  Run:  pip install -r requirements.txt")

# ── Download all model + tokenizer files in one call ─────────────────────────
print("Downloading model snapshot…  (this may take a minute on first run)")
snapshot_download(
    repo_id=HF_MODEL_ID,
    local_dir=str(dest),
    token=HF_TOKEN,
    # Exclude large ONNX variants and the unquantised "original/" checkpoint
    # — we only need the top-level safetensors weights for inference.
    ignore_patterns=["*.onnx", "*.onnx_data", "onnx/*", "original/*"],
)

print()
print("✓ Model files saved:")
for f in sorted(dest.rglob("*")):
    if f.is_file():
        size_mb = f.stat().st_size / 1_048_576
        print(f"    {f.relative_to(dest)}  ({size_mb:.1f} MB)")

print()
print("=" * 60)
print("  Next steps")
print("=" * 60)
print(f"  1. Copy .env.example → .env  (if you haven't already)")
print(f"  2. Set  LOCAL_MODEL_DIR={LOCAL_MODEL_DIR}")
print(f"  3. Set  TRANSFORMERS_OFFLINE=1")
print(f"  4. Run  python demo.py")
print("=" * 60)
