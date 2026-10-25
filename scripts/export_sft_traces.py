#!/usr/bin/env python3
"""Export SFT training traces from scripted curator prediction runs.

Reads predictions.jsonl from one or more run directories, extracts turn-level
conversation traces (probe → retrieve → answer) from scripted_curator results,
and writes a combined JSONL for SFT cold-start of the LoRA-GRPO stage.

Only correct predictions are included by default (--all to include all).

Usage:
    python scripts/export_sft_traces.py \\
        --runs runs/gpu_nextgqa_qwen2_5_full runs/gpu_nextqa_qwen2_5_full \\
        --qa-paths data/processed/nextgqa_val.jsonl data/processed/nextqa_val.jsonl \\
        --output data/rl/vidcurator_train_sft.jsonl

Output format (one JSON per line):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user",   "content": "..."},
            {"role": "assistant", "content": "{\"action\": \"probe\", ...}"}
        ],
        "metadata": {"qid": "...", "hop": 1, "action": "probe", "policy": "scripted_curator"}
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow running from project root without installing
sys.path.insert(0, str(Path(__file__).parent.parent))

from vidcurator.grpo import export_sft_dataset
from vidcurator.io import load_examples, read_jsonl
from vidcurator.types import PolicyResult


def main() -> int:
    ap = argparse.ArgumentParser(description="Export SFT traces from scripted curator runs.")
    ap.add_argument(
        "--runs", nargs="+", required=True,
        help="Run directories containing predictions.jsonl",
    )
    ap.add_argument(
        "--qa-paths", nargs="+", required=True,
        help="QA JSONL files (one per run, or a single shared file)",
    )
    ap.add_argument(
        "--output", default="data/rl/vidcurator_train_sft.jsonl",
        help="Output JSONL path",
    )
    ap.add_argument(
        "--policy", default="scripted_curator",
        help="Policy to extract traces from (default: scripted_curator)",
    )
    ap.add_argument(
        "--all", dest="include_all", action="store_true",
        help="Include incorrect predictions (default: correct-only)",
    )
    ap.add_argument(
        "--budget", type=int, default=None,
        help="Only extract traces from this budget (default: all budgets)",
    )
    ap.add_argument(
        "--val-output", default=None,
        help="If set, hold out a validation split to this path (split by video_id)",
    )
    ap.add_argument(
        "--val-frac", type=float, default=0.1,
        help="Fraction of videos held out for validation (default: 0.1)",
    )
    ap.add_argument(
        "--split-seed", type=int, default=13,
        help="Seed for the deterministic video-level train/val split",
    )
    args = ap.parse_args()

    qa_paths = args.qa_paths
    if len(qa_paths) == 1 and len(args.runs) > 1:
        qa_paths = qa_paths * len(args.runs)
    if len(qa_paths) != len(args.runs):
        print(f"ERROR: --qa-paths must be 1 or match --runs count ({len(args.runs)})")
        return 1

    import hashlib
    from vidcurator.grpo import extract_sft_traces

    def is_val(video_id: str) -> bool:
        """Deterministic per-video train/val assignment (video-level, no leakage)."""
        if not args.val_output or args.val_frac <= 0:
            return False
        digest = hashlib.md5(f"{args.split_seed}:{video_id}".encode()).hexdigest()
        bucket = int(digest[:8], 16) / 0xFFFFFFFF
        return bucket < args.val_frac

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    val_output = Path(args.val_output) if args.val_output else None
    if val_output:
        val_output.parent.mkdir(parents=True, exist_ok=True)

    train_rows = 0
    val_rows = 0
    train_fh = open(output, "w", encoding="utf-8")
    val_fh = open(val_output, "w", encoding="utf-8") if val_output else None
    try:
        for run_dir, qa_path in zip(args.runs, qa_paths):
            pred_path = Path(run_dir) / "predictions.jsonl"
            if not pred_path.exists():
                print(f"WARN: {pred_path} not found, skipping")
                continue
            if not Path(qa_path).exists():
                print(f"WARN: {qa_path} not found, skipping")
                continue

            examples = load_examples(qa_path)
            by_qid = {ex.qid: ex for ex in examples}
            raw_preds = read_jsonl(pred_path)

            run_train = 0
            run_val = 0
            for raw in raw_preds:
                if raw.get("policy") != args.policy:
                    continue
                if args.budget is not None and raw.get("budget") != args.budget:
                    continue
                result = PolicyResult.from_dict(raw)
                example = by_qid.get(result.qid)
                if example is None:
                    continue
                target_fh = val_fh if (val_fh and is_val(result.video_id)) else train_fh
                for trace in extract_sft_traces(result, example, only_correct=not args.include_all):
                    target_fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
                    if target_fh is val_fh:
                        run_val += 1
                        val_rows += 1
                    else:
                        run_train += 1
                        train_rows += 1
            print(f"  {Path(run_dir).name}: train+={run_train} val+={run_val}")
    finally:
        train_fh.close()
        if val_fh:
            val_fh.close()

    print(f"Wrote {train_rows} train rows to {output}")
    if val_output:
        print(f"Wrote {val_rows} val rows to {val_output}")
    return 0 if train_rows > 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
