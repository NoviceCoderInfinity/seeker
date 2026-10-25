#!/usr/bin/env python3
"""SF-Δ + E-D: merge best_single_frame runs (runs/ed_bsf_<ds>) with the R1 ablation runs.

SF-Δ (single-frame delta) = policy accuracy − best_single_frame accuracy, over the shared
qids, per policy@budget per dataset. Positive SF-Δ = multi-frame context genuinely helps.

E-D (single-frame solvability audit) = best_single_frame accuracy per dataset = "% solvable
with ≤1 frame". (A text-only/0-frame row would give "% solvable with none" — not run here.)

Both were impossible before because best_single_frame was absent from every run; we ran it
separately (capped candidate scan, budget=1) so it must be merged post-hoc.

Usage: python scripts/compute_sf_delta_ed.py \
         --datasets nextgqa nextqa egoschema mlvu longvideobench videomme \
         --out-md reports/sf_delta_ed.md --out-csv reports/sf_delta_ed.csv
"""
from __future__ import annotations
import argparse, csv, json
from pathlib import Path


def gold_for(config_path):
    cfg = json.loads(Path(config_path).read_text())
    qa = cfg["qa_path"]
    return {q["qid"]: q.get("answer_index") for q in map(json.loads, open(qa))}


def correctness(preds_path, gold, policy=None):
    """qid -> correct(bool) for a given policy (or all if policy=None), latest wins."""
    out = {}
    for line in open(preds_path):
        r = json.loads(line)
        if policy and r.get("policy") != policy:
            continue
        g = gold.get(r.get("qid"))
        ok = (g is not None and r.get("answer_index") is not None
              and int(r["answer_index"]) == int(g))
        out[r["qid"]] = ok
    return out


def correctness_by_group(preds_path, gold):
    """(policy, budget) -> {qid: correct}."""
    groups = {}
    for line in open(preds_path):
        r = json.loads(line)
        key = (r.get("policy"), int(r.get("budget", 0)))
        g = gold.get(r.get("qid"))
        ok = (g is not None and r.get("answer_index") is not None
              and int(r["answer_index"]) == int(g))
        groups.setdefault(key, {})[r["qid"]] = ok
    return groups


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+",
                    default=["nextgqa", "nextqa", "egoschema", "mlvu", "longvideobench", "videomme"])
    ap.add_argument("--policies", nargs="+",
                    default=["uniform", "aks", "scripted_curator", "curator_siglip", "videotree", "mmr_retrieve"])
    ap.add_argument("--out-md", default="reports/sf_delta_ed.md")
    ap.add_argument("--out-csv", default="reports/sf_delta_ed.csv")
    args = ap.parse_args()

    rows = []          # SF-Δ rows
    ed_rows = []       # E-D per-dataset bsf accuracy
    for ds in args.datasets:
        bsf_run = Path(f"runs/ed_bsf_{ds}/predictions.jsonl")
        r1_run = Path(f"runs/r1_v2_{ds}_qwen2_5/predictions.jsonl")
        bsf_cfg = f"configs/ed_bsf_{ds}.json"
        r1_cfg = f"configs/r1_v2_{ds}_qwen2_5.json"
        if not bsf_run.exists():
            print(f"  … {ds}: bsf run not done yet, skipping")
            continue
        if not r1_run.exists():
            print(f"  ! {ds}: R1 predictions missing, skipping")
            continue
        gold = gold_for(bsf_cfg)
        bsf_ok = correctness(bsf_run, gold, policy="best_single_frame")
        bsf_acc = sum(bsf_ok.values()) / len(bsf_ok) if bsf_ok else 0.0
        ed_rows.append({"dataset": ds, "bsf_acc": round(bsf_acc, 4), "n": len(bsf_ok)})
        r1_gold = gold_for(r1_cfg)
        groups = correctness_by_group(r1_run, r1_gold)
        for (pol, bud), qmap in sorted(groups.items()):
            if pol not in args.policies:
                continue
            shared = set(qmap) & set(bsf_ok)
            if not shared:
                continue
            pol_acc = sum(qmap[q] for q in shared) / len(shared)
            sf_acc = sum(bsf_ok[q] for q in shared) / len(shared)
            rows.append({"dataset": ds, "policy": pol, "budget": bud, "n": len(shared),
                         "policy_acc": round(pol_acc, 4), "bsf_acc": round(sf_acc, 4),
                         "sf_delta": round(pol_acc - sf_acc, 4)})

    if not ed_rows:
        print("no bsf runs finished yet — nothing to merge"); return 1

    # CSV
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)

    # Markdown
    lines = ["# SF-Δ + E-D (single-frame solvability & multi-frame value)", "",
             "## E-D — best-single-frame accuracy (% solvable with ≤1 frame)", "",
             "| dataset | n | bsf_acc |", "|---|---|---|"]
    for e in ed_rows:
        lines.append(f"| {e['dataset']} | {e['n']} | {e['bsf_acc']} |")
    lines += ["", "## SF-Δ — policy accuracy − best-single-frame (positive = multi-frame helps)",
              "", "| dataset | policy | K | n | policy_acc | bsf_acc | SF-Δ |",
              "|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['policy']} | {r['budget']} | {r['n']} | "
                     f"{r['policy_acc']} | {r['bsf_acc']} | {r['sf_delta']:+.4f} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_md} ({len(rows)} SF-Δ rows, {len(ed_rows)} datasets) + {args.out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
