#!/usr/bin/env python3
"""Upload BRT-BERT to https://huggingface.co/sablister/BRT-BERT.

Uploads ONLY `brt-bert-relevance/final/` — the trained CrossEncoder plus its
tokenizer and model card. The sibling `checkpoint-*` directories (~1.9 GB of
optimizer state, scheduler state, RNG state) are intentionally excluded; they
are useful for resuming training, not for using the model.

Prereqs:
    pip install -U huggingface_hub
    hf auth login              # token must have write scope
    hf auth whoami             # verify scope before running

Usage (from systs repo root):
    python upload_brt_bert.py

Re-running uploads only changed files (HF Hub deduplicates by content hash).
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ID = "sablister/BRT-BERT"
LOCAL_DIR = Path(__file__).resolve().parent / "brt-bert-relevance" / "final"
COMMIT_MESSAGE = "Initial BRT-BERT release: SciBERT-based scientific relevance cross-encoder"


def main() -> int:
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import HfHubHTTPError
    except ImportError:
        print("huggingface_hub not installed. Run: pip install -U huggingface_hub", file=sys.stderr)
        return 1

    if not LOCAL_DIR.exists():
        print(f"Model directory not found: {LOCAL_DIR}", file=sys.stderr)
        return 1

    files = sorted(p for p in LOCAL_DIR.iterdir() if p.is_file())
    total_bytes = sum(p.stat().st_size for p in files)
    print(f"Uploading {len(files)} files ({total_bytes / 1024 / 1024:.1f} MB total) "
          f"from {LOCAL_DIR} → {REPO_ID}")
    for p in files:
        size = p.stat().st_size
        if size > 1024 * 1024:
            print(f"  {p.name}  ({size / 1024 / 1024:.1f} MB)")
        else:
            print(f"  {p.name}  ({size} B)")

    api = HfApi()
    try:
        url = api.upload_folder(
            folder_path=str(LOCAL_DIR),
            repo_id=REPO_ID,
            repo_type="model",
            commit_message=COMMIT_MESSAGE,
            ignore_patterns=["checkpoint-*", "runs/*", "*.tmp", ".DS_Store"],
        )
    except HfHubHTTPError as exc:
        print(f"Upload failed: {exc}", file=sys.stderr)
        if "401" in str(exc) or "403" in str(exc):
            print("Hint: your HF token may lack write scope. "
                  "Run `hf auth whoami` to check, or regenerate at "
                  "https://huggingface.co/settings/tokens", file=sys.stderr)
        return 1

    print(f"\nUploaded: {url}")
    print(f"View at: https://huggingface.co/{REPO_ID}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
