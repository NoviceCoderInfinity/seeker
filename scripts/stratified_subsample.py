#!/usr/bin/env python3
"""Stratified 50% (configurable) subsample of a QA JSONL.

Groups rows by the best available question-type key so every type is covered
proportionally, then samples a fraction from each stratum with a fixed seed
(reproducible). Guarantees >=1 row per stratum. Preserves all fields.

Usage:
  stratified_subsample.py --input full.jsonl --output half.jsonl --frac 0.5 --seed 0
"""
from __future__ import annotations
import argparse
import collections
import json
import math
import random
from pathlib import Path

TYPE_KEYS = ["question_type", "type", "task_type", "q_type", "sub_task"]


def type_of(row: dict) -> str:
    md = row.get("metadata", {}) or {}
    raw = md.get("raw", {}) or {}
    for k in TYPE_KEYS:
        for src in (row, md, raw):
            v = src.get(k)
            if v is not None:
                return f"{k}={v}"
    # fallback: keep answer-option balance
    return f"ans={row.get('answer_index')}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--frac", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.input) if l.strip()]
    strata: dict[str, list] = collections.defaultdict(list)
    for r in rows:
        strata[type_of(r)].append(r)

    rng = random.Random(args.seed)
    kept = []
    report = {}
    for key in sorted(strata):
        group = sorted(strata[key], key=lambda r: str(r.get("qid")))  # deterministic order
        n_keep = max(1, math.ceil(len(group) * args.frac))
        idx = sorted(rng.sample(range(len(group)), min(n_keep, len(group))))
        picked = [group[i] for i in idx]
        kept.extend(picked)
        report[key] = (len(group), len(picked))

    # preserve original file order among kept rows
    keep_ids = {id(r) for r in kept}
    ordered = [r for r in rows if id(r) in keep_ids]

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for r in ordered:
            fh.write(json.dumps(r) + "\n")

    print(f"{Path(args.input).name}: {len(rows)} -> {len(ordered)} ({100*len(ordered)/len(rows):.0f}%)")
    for key, (tot, kp) in report.items():
        print(f"   {key:32} {tot:5d} -> {kp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
