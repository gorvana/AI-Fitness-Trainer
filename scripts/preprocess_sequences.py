#!/usr/bin/env python3
"""
Convert dataset/sequences.jsonl into a pelvis-centered, smoothed version
saved as dataset/sequences_preprocessed.jsonl.

Usage:
  python3 scripts/preprocess_sequences.py
"""
import os
import sys
import logging

logger = logging.getLogger(__name__)

# Ensure project root is on sys.path so `from bot...` imports work when the script
# is executed directly (e.g. `python3 scripts/preprocess_sequences.py`).
SCRIPT_DIR = os.path.dirname(__file__)
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from bot.utils.sequence_preprocessing import load_jsonl, preprocess_record, save_jsonl


def main():
    src = os.path.join(os.getcwd(), "dataset", "sequences.jsonl")
    dst = os.path.join(os.getcwd(), "dataset", "sequences_preprocessed.jsonl")
    if not os.path.exists(src):
        logger.info(f"Source not found: {src}")
        return
    records = load_jsonl(src)
    out = []
    for r in records:
        out.append(preprocess_record(r, center="pelvis", scale_mode="hip_distance", ema_alpha=0.15))
    save_jsonl(dst, out)
    logger.info(f"Wrote {len(out)} records to {dst}")


if __name__ == "__main__":
    main()
