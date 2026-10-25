#!/usr/bin/env python3
"""E-J: Global adaptive budget allocation (pure post-hoc, no GPU).

Claim: fixed-K is the wrong contract. At a matched *average* frame budget, allocating
budget across questions by a difficulty signal beats giving every question the same K.

For one policy we load predictions across the budget sweep and, per question, record
correctness at each K and frames used. We then compare, at each average-frame level:
  - fixed-K:      every question gets the same K (the reported curve)
  - sufficiency:  allocate more budget to low-confidence questions (uses logged confidence
                  at the smallest budget as the "needs more frames" signal)
  - oracle:       greedy upper bound — spend the fewest frames that make each answerable
                  question correct (envelope; not achievable, shows the headroom)

Usage: python scripts/eval_budget_allocation.py \
    --preds runs/r1_v2_nextgqa_qwen2_5/predictions.jsonl --policy curator_siglip --qa data/processed/nextgqa_val_150.jsonl
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--policy", default="curator_siglip")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gold = {}
    for line in open(args.qa):
        q = json.loads(line)
        gold[q["qid"]] = q.get("answer_index")

    # per qid: {K: (correct, frames_used, confidence)}
    prof: dict[str, dict[int, tuple]] = defaultdict(dict)
    for line in open(args.preds):
        r = json.loads(line)
        if r.get("policy") != args.policy:
            continue
        qid = r.get("qid")
        if qid not in gold:
            continue
        K = int(r.get("budget", 0))
        correct = int(r.get("answer_index") == gold[qid])
        frames = float(r.get("num_frames", K))
        conf = float(r.get("confidence", 0.0))
        prof[qid][K] = (correct, frames, conf)

    budgets = sorted({K for p in prof.values() for K in p})
    qids = [q for q in prof if all(K in prof[q] for K in budgets)]
    if not qids:
        print(f"no complete question profiles for policy={args.policy}")
        return 1
    n = len(qids)

    # fixed-K curve
    fixedK = []
    for K in budgets:
        acc = sum(prof[q][K][0] for q in qids) / n
        avgf = sum(prof[q][K][1] for q in qids) / n
        fixedK.append((avgf, acc, K))

    # oracle: for a target avg budget, pick per-q the min-K that is correct (else min-K).
    # sweep target average across the budget grid.
    def alloc_curve(rank_key):
        # rank_key(qid) -> priority for extra budget (higher = give more frames first)
        curve = []
        order = sorted(qids, key=rank_key, reverse=True)
        for target_idx in range(len(budgets)):
            # give the top-(fraction) questions the larger budgets, rest the smallest
            # simple 2-level split at each target average is coarse; instead do a smooth
            # allocation: assign budget level by rank percentile matched to target average.
            hi = budgets[target_idx]
            correct = 0.0
            frames = 0.0
            # allocate hi to the neediest half..full depending on target index
            frac = (target_idx + 1) / len(budgets)
            cutoff = int(frac * n)
            for i, q in enumerate(order):
                K = hi if i < cutoff else budgets[0]
                correct += prof[q][K][0]
                frames += prof[q][K][1]
            curve.append((frames / n, correct / n, hi))
        return curve

    # sufficiency-adaptive: neediest = lowest confidence at smallest budget
    k0 = budgets[0]
    suff = alloc_curve(rank_key=lambda q: -prof[q][k0][2])

    # oracle upper bound: min frames to be correct
    oracle_pts = []
    for K in budgets:
        # each q: min frames among budgets<=K that yields correct, else frames at K
        correct = 0.0
        frames = 0.0
        for q in qids:
            best = None
            for kk in budgets:
                if kk > K:
                    break
                c, f, _ = prof[q][kk]
                if c:
                    best = f
                    break
            if best is None:
                best = prof[q][K][1]
                correct += prof[q][K][0]
            else:
                correct += 1
            frames += best
        oracle_pts.append((frames / n, correct / n, K))

    def show(name, pts):
        print(f"\n{name}")
        print(f"  {'avg_frames':>11}{'acc':>8}{'budget':>8}")
        for f, a, K in pts:
            print(f"  {f:>11.2f}{a:>8.3f}{K:>8}")

    print(f"=== E-J budget allocation | policy={args.policy} | n={n} | budgets={budgets} ===")
    show("fixed-K (reported contract)", fixedK)
    show("sufficiency-adaptive (matched avg)", suff)
    show("oracle allocation (upper bound)", oracle_pts)

    if args.out:
        json.dump({"policy": args.policy, "n": n, "fixedK": fixedK,
                   "sufficiency": suff, "oracle": oracle_pts}, open(args.out, "w"), indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
