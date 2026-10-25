#!/usr/bin/env python3
"""Convert VID-CURATOR SFT JSONL traces to verl-compatible Parquet for GRPO."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd


def _buffer_state(user_text: str) -> dict:
    """Parse budget + held-frame count from the rendered buffer in the last user turn.

    Efficiency-aware reward modes (outcome_eff) need the buffer occupancy at action time.
    """
    out: dict = {}
    m = re.search(r"Budget K=(\d+)", user_text)
    if m:
        out["budget"] = int(m.group(1))
    held = re.search(r"Held frames:\n((?:\s*- id=\S+\n?)+)", user_text)
    if held:
        out["frames_in_buffer"] = held.group(1).count("- id=")
    elif re.search(r"Held frames:\n\s*\(none\)", user_text):
        # Empty buffer renders as "(none)" — 0 held frames, not missing metadata.
        # Probe-turn shaping needs this state most (fill=1.0 for retrieve).
        out["frames_in_buffer"] = 0
    return out


def row_to_verl(record: dict) -> dict:
    messages = record["messages"]
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("expected final assistant message")
    prompt = messages[:-1]
    ground_truth = messages[-1]["content"]
    meta = dict(record.get("metadata") or {})
    for msg in reversed(prompt):
        if msg.get("role") == "user":
            meta.update(_buffer_state(str(msg.get("content", ""))))
            break
    return {
        "prompt": prompt,
        "data_source": "vidcurator",
        "ability": "action",
        "reward_model": {
            "style": "rule",
            "ground_truth": ground_truth,
        },
        "extra_info": meta,
    }


def convert_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(row_to_verl(json.loads(line)))
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", required=True)
    parser.add_argument("--val", required=True)
    parser.add_argument("--out-dir", default="data/rl/parquet")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    train_rows = convert_jsonl(Path(args.train))
    val_rows = convert_jsonl(Path(args.val))

    train_path = out_dir / "train.parquet"
    val_path = out_dir / "val.parquet"
    pd.DataFrame(train_rows).to_parquet(train_path, index=False)
    pd.DataFrame(val_rows).to_parquet(val_path, index=False)

    print(f"wrote {train_path} ({len(train_rows)} rows)")
    print(f"wrote {val_path} ({len(val_rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
