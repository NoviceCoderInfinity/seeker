#!/usr/bin/env python3
"""Score a learned_curator_sc shard as a pass@k potential probe.

Reads the sc_votes list (k sampled curation trajectories per question) stored in each
row's metadata by SelfConsistencyCuratorPolicy, and reports:
  - pass@j for j in 1,2,4,8,16,32 via the unbiased estimator
    pass@j = mean_q [ 1 - C(k-c_q, j) / C(k, j) ],  c_q = #correct votes for question q
  - greedy/majority-vote accuracy (what the shard's answer_index already encodes)
  - mean per-question correct-vote fraction (how concentrated the mass already is)

Interpretation: pass@32 >> pass@1 means the sampled policy already contains winning
trajectories that reward shaping / more RL epochs could select => headroom exists.
pass@32 ~= pass@1 means the setup is capped and no reward will fix it.

Usage: python scripts/passk_score.py --run runs/passk32_p4_nextgqa_b8 \
           --gold data/processed/heldout/nextgqa_heldout_p120.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os


def unbiased_pass_at_j(k: int, c: int, j: int) -> float:
    if c == 0:
        return 0.0
    if k - c < j:
        return 1.0
    return 1.0 - math.comb(k - c, j) / math.comb(k, j)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--gold", required=True)
    ap.add_argument("--ks", default="1,2,4,8,16,32")
    args = ap.parse_args()

    gold = {}
    for line in open(args.gold):
        q = json.loads(line)
        gold[q["qid"]] = int(q["answer_index"])

    shards = sorted(glob.glob(os.path.join(args.run, "_shards", "*learned_curator_sc*.jsonl")))
    if not shards:
        raise SystemExit(f"no learned_curator_sc shards under {args.run}/_shards")

    for sh in shards:
        rows = [json.loads(l) for l in open(sh)]
        per_q = []  # (k, n_correct_votes, majority_correct)
        for r in rows:
            qid = r["qid"]
            if qid not in gold:
                continue
            votes = (r.get("metadata") or {}).get("sc_votes")
            if votes is None:
                continue
            c = sum(1 for v in votes if v is not None and int(v) == gold[qid])
            maj_ok = r.get("answer_index") is not None and int(r["answer_index"]) == gold[qid]
            per_q.append((len(votes), c, maj_ok))
        if not per_q:
            print(f"{sh}: no scorable rows (metadata.sc_votes missing?)")
            continue
        n = len(per_q)
        kmax = min(k for k, _, _ in per_q)
        print(f"\n== {os.path.basename(sh)}  n={n}  k={kmax} ==")
        for j in (int(x) for x in args.ks.split(",")):
            if j > kmax:
                continue
            p = sum(unbiased_pass_at_j(k, c, j) for k, c, _ in per_q) / n
            print(f"  pass@{j:<3d} = {p:.3f}")
        maj = sum(1 for _, _, m in per_q if m) / n
        frac = sum(c / k for k, c, _ in per_q) / n
        solved_any = sum(1 for _, c, _ in per_q if c > 0) / n
        print(f"  majority-vote acc = {maj:.3f}")
        print(f"  mean correct-vote fraction = {frac:.3f}")
        print(f"  questions with >=1 correct trajectory = {solved_any:.3f}")


if __name__ == "__main__":
    main()
