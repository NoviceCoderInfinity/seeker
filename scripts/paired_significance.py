#!/usr/bin/env python3
"""Paired significance tests for two policies on the SAME questions (plan5.md §7, HANDOFF STEP 7).

Two tests, both paired on qid so they exploit the controlled harness (same VLM/prompt/pool; only the
policy differs):
  1. McNemar (per budget) on accuracy — exact two-sided binomial on the discordant pairs.
  2. Paired bootstrap on CE-AUC (budget-averaged accuracy) — resample qids with replacement, recompute
     the CE-AUC difference A-B, report the mean, 95% CI, and a two-sided bootstrap p.

Correctness = predicted answer_index (from runs/<dir>/_shards/b<budget>_<policy>.jsonl)
              == gold answer_index (from the QA file), joined on qid. No re-inference; pure post-hoc.

Usage:
  # curator vs uniform inside one headline run-dir:
  python scripts/paired_significance.py --run runs/fs_nextgqa_qwen2_5_vl_7b_headline \
      --a scripted_curator --b uniform
  # learned (separate run-dir) vs uniform (headline run-dir):
  python scripts/paired_significance.py \
      --run-a runs/fs_videomme_qwen2_5_vl_7b_learned --a learned_curator \
      --run-b runs/fs_videomme_qwen2_5_vl_7b_headline --b uniform \
      --out reports/fs_sig_videomme_learned_vs_uniform.md

QA gold file is read from each run's summary.json (config.qa_path) unless --qa is given.
Budgets default to the intersection of both policies' available shards.
"""
import argparse
import json
import math
from pathlib import Path

import numpy as np


def load_gold(qa_path):
    gold = {}
    with open(qa_path) as f:
        for line in f:
            r = json.loads(line)
            gold[str(r["qid"])] = int(r["answer_index"])
    return gold


def load_correct(run_dir, policy, budget, gold):
    """qid -> 1/0 correctness for one (policy, budget) shard."""
    shard = Path(run_dir) / "_shards" / f"b{budget}_{policy}.jsonl"
    if not shard.exists():
        return None
    out = {}
    with open(shard) as f:
        for line in f:
            r = json.loads(line)
            qid = str(r["qid"])
            if qid in gold and r.get("answer_index") is not None:
                out[qid] = int(int(r["answer_index"]) == gold[qid])
    return out


def qa_from_summary(run_dir):
    s = json.loads((Path(run_dir) / "summary.json").read_text())
    return s["config"]["qa_path"], s["config"]["budgets"]


def available_budgets(run_dir, policy):
    sd = Path(run_dir) / "_shards"
    return sorted(int(p.name[1:].split("_", 1)[0])
                  for p in sd.glob(f"b*_{policy}.jsonl")) if sd.exists() else []


def mcnemar_exact_p(b10, b01):
    """Two-sided exact binomial McNemar p over discordant pairs (log-space, no scipy)."""
    n = b10 + b01
    if n == 0:
        return 1.0
    k = min(b10, b01)
    # P(X<=k) under Binomial(n,0.5); two-sided = 2*that, capped at 1.
    logp = []
    for i in range(0, k + 1):
        lc = math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1)
        logp.append(lc + n * math.log(0.5))
    m = max(logp)
    tail = math.exp(m) * sum(math.exp(x - m) for x in logp)
    return min(1.0, 2.0 * tail)


def paired_bootstrap_ceauc(corr_a, corr_b, budgets, reps=10000, seed=0):
    """Resample shared qids; CE-AUC = mean over budgets of accuracy; return diff A-B stats."""
    shared = sorted(set.intersection(*[set(corr_a[bd]) & set(corr_b[bd]) for bd in budgets]))
    if not shared:
        return None
    A = np.array([[corr_a[bd][q] for q in shared] for bd in budgets], dtype=float)  # [n_bud, n_q]
    B = np.array([[corr_b[bd][q] for q in shared] for bd in budgets], dtype=float)
    rng = np.random.default_rng(seed)
    n = len(shared)
    diffs = np.empty(reps)
    for r in range(reps):
        idx = rng.integers(0, n, n)
        diffs[r] = A[:, idx].mean(axis=1).mean() - B[:, idx].mean(axis=1).mean()
    obs = A.mean(axis=1).mean() - B.mean(axis=1).mean()
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    frac_gt = float((diffs > 0).mean())
    p = 2.0 * min(frac_gt, 1.0 - frac_gt)
    return dict(n=n, obs=float(obs), ci_lo=float(lo), ci_hi=float(hi), p=float(min(1.0, p)),
                ceauc_a=float(A.mean(axis=1).mean()), ceauc_b=float(B.mean(axis=1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", help="single run-dir holding both policies")
    ap.add_argument("--run-a"); ap.add_argument("--run-b")
    ap.add_argument("--a", required=True, help="policy A (e.g. scripted_curator / learned_curator)")
    ap.add_argument("--b", required=True, help="policy B (e.g. uniform)")
    ap.add_argument("--qa", default="", help="gold QA jsonl (default: from summary.json)")
    ap.add_argument("--budgets", default="", help="comma list; default = intersection of both")
    ap.add_argument("--reps", type=int, default=10000)
    ap.add_argument("--out", default="", help="write a markdown report here as well as stdout")
    args = ap.parse_args()

    run_a = args.run_a or args.run
    run_b = args.run_b or args.run
    if not run_a or not run_b:
        raise SystemExit("give --run (both policies) or --run-a and --run-b")

    qa = args.qa or qa_from_summary(run_a)[0]
    gold = load_gold(qa)

    if args.budgets:
        budgets = [int(x) for x in args.budgets.split(",")]
    else:
        budgets = sorted(set(available_budgets(run_a, args.a)) & set(available_budgets(run_b, args.b)))
    if not budgets:
        raise SystemExit(f"no common budgets for {args.a}@{run_a} and {args.b}@{run_b}")

    corr_a = {bd: load_correct(run_a, args.a, bd, gold) for bd in budgets}
    corr_b = {bd: load_correct(run_b, args.b, bd, gold) for bd in budgets}
    for bd in budgets:
        if corr_a[bd] is None or corr_b[bd] is None:
            raise SystemExit(f"missing shard at budget {bd} (a={corr_a[bd] is not None}, b={corr_b[bd] is not None})")

    lines = [f"# Paired significance — {args.a} vs {args.b}",
             f"_run-a=`{run_a}` · run-b=`{run_b}` · qa=`{qa}` · reps={args.reps}_", "",
             "## McNemar per budget (exact two-sided binomial on discordant pairs)", "",
             "| budget | n | acc_A | acc_B | Δacc | A✓B✗ | A✗B✓ | p | sig |",
             "|---|---|---|---|---|---|---|---|---|"]
    for bd in budgets:
        ca, cb = corr_a[bd], corr_b[bd]
        shared = sorted(set(ca) & set(cb))
        n = len(shared)
        accA = sum(ca[q] for q in shared) / n
        accB = sum(cb[q] for q in shared) / n
        b10 = sum(1 for q in shared if ca[q] == 1 and cb[q] == 0)  # A right, B wrong
        b01 = sum(1 for q in shared if ca[q] == 0 and cb[q] == 1)
        p = mcnemar_exact_p(b10, b01)
        sig = "SIG" if p < 0.05 else "ns"
        lines.append(f"| {bd} | {n} | {accA:.3f} | {accB:.3f} | {accA-accB:+.3f} | {b10} | {b01} | {p:.4f} | {sig} |")

    bs = paired_bootstrap_ceauc(corr_a, corr_b, budgets, reps=args.reps)
    lines += ["", "## Paired bootstrap on CE-AUC (budget-averaged accuracy)", ""]
    if bs:
        sig = "SIG" if bs["p"] < 0.05 else "ns"
        lines += [f"- CE-AUC A = **{bs['ceauc_a']:.3f}** · CE-AUC B = **{bs['ceauc_b']:.3f}** · "
                  f"Δ = **{bs['obs']:+.3f}**",
                  f"- 95% CI [{bs['ci_lo']:+.3f}, {bs['ci_hi']:+.3f}] · two-sided p = **{bs['p']:.4f}** "
                  f"({sig}) · paired on n={bs['n']} shared qids over budgets {budgets}"]
    else:
        lines.append("- no shared qids across all budgets — cannot bootstrap")

    report = "\n".join(lines)
    print(report)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(report + "\n")
        print(f"\n[wrote {args.out}]")


if __name__ == "__main__":
    main()
