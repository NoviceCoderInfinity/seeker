#!/usr/bin/env python3
"""Summarize verl GRPO training logs into CSV + markdown for paper / debugging."""

from __future__ import annotations

import argparse
import csv
import re
import statistics
from pathlib import Path


VAL_RE = re.compile(
    r"step:(\d+).*?"
    r"val-core/vidcurator/reward/mean@1:([0-9.]+)"
    r".*?"
    r"val-aux/vidcurator/format_valid/mean@1:([0-9.]+)"
    r".*?"
    r"val-aux/vidcurator/action_match/mean@1:([0-9.]+)"
    r".*?"
    r"val-aux/vidcurator/field_match/mean@1:([0-9.]+)"
)
TRAIN_RE = re.compile(r"step:(\d+).*?critic/rewards/mean:([0-9.]+)")


def parse_log(path: Path) -> tuple[list[dict], list[dict]]:
    val_rows: list[dict] = []
    train_rows: list[dict] = []
    seen_val: set[int] = set()
    seen_train: set[int] = set()
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = VAL_RE.search(line)
            if m:
                step = int(m.group(1))
                if step not in seen_val:
                    seen_val.add(step)
                    val_rows.append(
                        {
                            "step": step,
                            "val_reward": float(m.group(2)),
                            "format_valid": float(m.group(3)),
                            "action_match": float(m.group(4)),
                            "field_match": float(m.group(5)),
                        }
                    )
            m = TRAIN_RE.search(line)
            if m:
                step = int(m.group(1))
                if step not in seen_train:
                    seen_train.add(step)
                    train_rows.append({"step": step, "train_reward": float(m.group(2))})
    val_rows.sort(key=lambda r: r["step"])
    train_rows.sort(key=lambda r: r["step"])
    return val_rows, train_rows


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_summary(job_id: str, val_rows: list[dict], train_rows: list[dict]) -> str:
    lines = [f"## GRPO job {job_id}", ""]
    if train_rows:
        rewards = [r["train_reward"] for r in train_rows]
        lines += [
            f"- Train steps logged: **{len(train_rows)}** (step {train_rows[0]['step']}–{train_rows[-1]['step']})",
            f"- Train reward mean: **{statistics.mean(rewards):.4f}** (σ={statistics.stdev(rewards) if len(rewards)>1 else 0:.4f})",
            f"- Train reward range: [{min(rewards):.4f}, {max(rewards):.4f}]",
            "",
        ]
    if val_rows:
        best = max(val_rows, key=lambda r: r["val_reward"])
        last = val_rows[-1]
        lines += [
            "| step | val_reward | action_match | field_match | format_valid |",
            "|---:|---:|---:|---:|---:|",
        ]
        for row in val_rows:
            lines.append(
                f"| {row['step']} | {row['val_reward']:.4f} | {row['action_match']:.4f} | "
                f"{row['field_match']:.4f} | {row['format_valid']:.4f} |"
            )
        lines += [
            "",
            f"- Best val reward: **{best['val_reward']:.4f}** @ step {best['step']}",
            f"- Last val checkpoint: reward **{last['val_reward']:.4f}**, action_match **{last['action_match']:.4f}**",
            "",
        ]
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--log", action="append", required=True, help="grpo_train_<jobid>.log path")
    parser.add_argument("--out-dir", default="reports")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    sections: list[str] = ["# GRPO training summary", ""]

    for log_arg in args.log:
        log_path = Path(log_arg)
        job_id = log_path.stem.replace("grpo_train_", "")
        val_rows, train_rows = parse_log(log_path)
        write_csv(out_dir / f"grpo_train_{job_id}.csv", train_rows, ["step", "train_reward"])
        write_csv(out_dir / f"grpo_val_{job_id}.csv", val_rows, ["step", "val_reward", "action_match", "field_match", "format_valid"])
        sections.append(markdown_summary(job_id, val_rows, train_rows))

    md_path = out_dir / "grpo_summary.md"
    md_path.write_text("\n".join(sections) + "\n", encoding="utf-8")
    print(f"Wrote {md_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
