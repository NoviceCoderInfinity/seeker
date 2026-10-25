#!/usr/bin/env python3
"""Aggregate all runs/*/summary.json into paper-ready tables.

Outputs (under --out-dir, default reports/):
  - all_results.csv          long form: run,dataset,backbone,split,policy,budget,n,acc,ci_lo,ci_hi,frames,fep,fer,if,pc_ece
  - pareto_<dataset>.md       accuracy x budget per policy (the main Pareto tables)
  - ce_auc.csv               context-efficiency AUC per run x policy
  - sf_delta.csv             single-frame-shortcut gap per run x policy@budget
  - recall_at_k.csv          retrieval recall@K per run
  - summary.md               human-readable headline (curator vs uniform per dataset/budget)

Usage: python scripts/aggregate_results.py [--glob 'runs/gpu_*']
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from collections import defaultdict
from pathlib import Path


def wilson_ci(acc: float, n: int, z: float = 1.96):
    """95% Wilson score interval (no per-example data needed)."""
    if not n:
        return (None, None)
    p = acc
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt((p * (1 - p) + z * z / (4 * n)) / n)) / denom
    return (round(max(0.0, center - half), 4), round(min(1.0, center + half), 4))


def parse_run_name(name: str):
    """gpu_nextgqa_qwen2_5_full -> (nextgqa, qwen2_5, full)."""
    s = name[4:] if name.startswith("gpu_") else name
    backbone, split = "?", "?"
    for b in ["qwen2_5", "qwen3", "internvl3_5", "llava_video", "qwen3_thinking"]:
        if f"_{b}_" in f"_{s}_":
            backbone = b
            s2 = s.replace(f"_{b}", "")
            break
    else:
        s2 = s
    for sp in [
        "learned_grpo_analysis",
        "learned_grpo",
        "full_siglip",
        "siglip_analysis",
        "full_80pct_invalid",
        "zero_shot",
        "full",
        "analysis",
        "test",
        "val",
    ]:
        if s2.endswith(f"_{sp}"):
            split = sp
            s2 = s2[: -(len(sp) + 1)]
            break
    dataset = s2.strip("_")
    return dataset, backbone, split


def load_runs(pattern: str, exclude: str = ""):
    runs = {}
    excludes = [e for e in exclude.split(",") if e]
    for pat in pattern.split(","):
        for sm in glob.glob(f"{pat}/summary.json"):
            run = Path(sm).parent.name
            if any(e in run for e in excludes):
                continue
            try:
                runs[run] = json.load(open(sm))
            except Exception as exc:
                print(f"skip {run}: {exc}")
    return runs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/gpu_*")
    ap.add_argument("--exclude", default="", help="comma-separated run-name substrings to skip (e.g. v1 zero_shot runs)")
    ap.add_argument("--out-dir", default="reports")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    runs = load_runs(args.glob, args.exclude)
    if not runs:
        print(f"no runs matched {args.glob}/summary.json")
        return 1
    print(f"loaded {len(runs)} runs")

    # 1) long CSV
    rows = []
    for run, s in sorted(runs.items()):
        ds, bb, sp = parse_run_name(run)
        for key, g in s.get("groups", {}).items():
            n = int(g.get("n", 0))
            acc = float(g.get("accuracy", 0.0))
            # Use pre-computed Wilson CI from summary if available, else compute
            ci_lo = g.get("ci_lo")
            ci_hi = g.get("ci_hi")
            if ci_lo is None or ci_hi is None:
                ci_lo, ci_hi = wilson_ci(acc, n)
            rows.append({
                "run": run, "dataset": ds, "backbone": bb, "split": sp,
                "policy": g.get("policy"), "budget": g.get("budget"), "n": n,
                "acc": round(acc, 4), "ci_lo": ci_lo, "ci_hi": ci_hi,
                "avg_frames": round(float(g.get("avg_frames", 0)), 2),
                "avg_visual_tokens_est": round(float(g.get("avg_visual_tokens_est", 0)), 1),
                "avg_probes": round(float(g.get("avg_probes", 0)), 3),
                "avg_latency_s": g.get("avg_latency_s"),
                "fep": g.get("fep"), "fer": g.get("fer"), "if": g.get("if"),
                "pc_ece": g.get("pc_ece"),
            })
    with open(out / "all_results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # 1b) cost table CSV
    cost_rows = []
    for run, s in sorted(runs.items()):
        ds, bb, sp = parse_run_name(run)
        for key, ct in (s.get("cost_table") or {}).items():
            cost_rows.append({
                "run": run, "dataset": ds, "backbone": bb, "split": sp,
                "policy_budget": key,
                "avg_frames": ct.get("avg_frames"),
                "avg_visual_tokens_est": ct.get("avg_visual_tokens_est"),
                "avg_latency_s": ct.get("avg_latency_s"),
            })
    if cost_rows:
        with open(out / "cost_table.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(cost_rows[0].keys()))
            w.writeheader()
            w.writerows(cost_rows)

    # 2) per-dataset Pareto md (acc x budget per policy), one table per (dataset,backbone,split)
    by_grp = defaultdict(lambda: defaultdict(dict))  # (ds,bb,sp) -> policy -> budget -> (acc,n)
    budgets_seen = defaultdict(set)
    for r in rows:
        gkey = (r["dataset"], r["backbone"], r["split"])
        by_grp[gkey][r["policy"]][r["budget"]] = (r["acc"], r["n"])
        budgets_seen[gkey].add(r["budget"])
    pareto_md = ["# Pareto tables (accuracy vs frame budget)\n"]
    for gkey, pol_map in sorted(by_grp.items()):
        ds, bb, sp = gkey
        buds = sorted(b for b in budgets_seen[gkey] if b is not None)
        pareto_md.append(f"\n## {ds} | {bb} | {sp}\n")
        pareto_md.append("| policy | " + " | ".join(f"@{b}" for b in buds) + " |")
        pareto_md.append("|" + "---|" * (len(buds) + 1))
        for pol in sorted(pol_map):
            cells = []
            for b in buds:
                v = pol_map[pol].get(b)
                cells.append(f"{v[0]:.3f}" if v else "—")
            pareto_md.append(f"| {pol} | " + " | ".join(cells) + " |")
    (out / "pareto_all.md").write_text("\n".join(pareto_md) + "\n")

    # 3) CE-AUC, 4) SF-delta, 5) recall@K
    with open(out / "ce_auc.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "policy", "ce_auc"])
        for run, s in sorted(runs.items()):
            for pol, v in (s.get("context_efficiency") or {}).items():
                w.writerow([run, pol, round(v, 4) if v is not None else ""])
    with open(out / "sf_delta.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "policy_budget", "sf_delta"])
        for run, s in sorted(runs.items()):
            for k, v in (s.get("sf_delta") or {}).items():
                w.writerow([run, k, round(v, 4) if v is not None else ""])

    # SF-Δ by question type
    with open(out / "sf_delta_by_type.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "policy_budget", "question_type", "sf_delta"])
        for run, s in sorted(runs.items()):
            for pb_key, type_map in (s.get("sf_delta_by_type") or {}).items():
                for qtype, v in (type_map or {}).items():
                    w.writerow([run, pb_key, qtype, round(v, 4) if v is not None else ""])
    with open(out / "recall_at_k.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["run", "k", "recall"])
        for run, s in sorted(runs.items()):
            for k, v in (s.get("retrieval_recall_at_k") or {}).items():
                w.writerow([run, k, round(v, 4) if v is not None else ""])

    # 6) headline summary.md: curator vs uniform + FlexFrame + baselines (with 95% Wilson CI)
    # NOTE: curator_oracle_probe is deliberately excluded — it reads gold spans
    # (analysis ceiling only) and must never appear in headline comparison tables.
    HEADLINE_POLICIES = ["scripted_curator", "curator_siglip", "flexframe", "aks", "videotree", "uniform"]

    # Build a per-(run,policy,budget) mapping with CI info for the headline table
    # rows is the long CSV already built above
    ci_lookup: dict = {}
    for r in rows:
        gkey = (r["dataset"], r["backbone"], r["split"])
        ci_lookup[(gkey, r["policy"], r["budget"])] = {
            "acc": r["acc"], "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"], "n": r["n"]
        }

    head = ["# Headline: curator vs baselines (accuracy ± 95% Wilson CI)\n"]
    for gkey, pol_map in sorted(by_grp.items()):
        ds, bb, sp = gkey
        present = [p for p in HEADLINE_POLICIES if p in pol_map]
        if not present:
            continue
        head.append(f"\n## {ds} | {bb} | {sp}\n")
        header_cols = " | ".join(f"@{b}" for b in sorted(x for x in budgets_seen[gkey] if x is not None))
        head.append(f"| policy | {header_cols} |")
        head.append("|" + "---|" * (len(budgets_seen[gkey]) + 1))
        for pol in present:
            cells = []
            for b in sorted(x for x in budgets_seen[gkey] if x is not None):
                info = ci_lookup.get((gkey, pol, b))
                if info:
                    lo = f"{info['ci_lo']:.3f}" if info["ci_lo"] is not None else "?"
                    hi = f"{info['ci_hi']:.3f}" if info["ci_hi"] is not None else "?"
                    cells.append(f"{info['acc']:.3f} [{lo},{hi}]")
                else:
                    cells.append("—")
            head.append(f"| {pol} | " + " | ".join(cells) + " |")

        # CE-AUC sub-table for this group
        ce_rows = [(run, s) for run, s in runs.items() if parse_run_name(run) == (ds, bb, sp)]
        if ce_rows:
            _, ce_s = ce_rows[0]
            ce = ce_s.get("context_efficiency") or {}
            if ce:
                head.append("\n**CE-AUC (context efficiency):**\n")
                head.append("| policy | CE-AUC |")
                head.append("|---|---|")
                for pol in sorted(ce, key=lambda p: HEADLINE_POLICIES.index(p) if p in HEADLINE_POLICIES else 99):
                    v = ce[pol]
                    head.append(f"| {pol} | {v:.4f} |" if v is not None else f"| {pol} | — |")

    (out / "summary.md").write_text("\n".join(head) + "\n")

    print(
        f"wrote {out}/all_results.csv, pareto_all.md, ce_auc.csv, "
        f"sf_delta.csv, sf_delta_by_type.csv, recall_at_k.csv, cost_table.csv, summary.md"
    )
    print(f"  rows: {len(rows)} | runs: {len(runs)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
