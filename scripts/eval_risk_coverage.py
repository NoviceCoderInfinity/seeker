#!/usr/bin/env python3
"""E-C: selective prediction via the curator's SUFFICIENCY signal.

NOTE: the answerer's `confidence` field is a constant placeholder (0.5) — it carries no
signal, so the stored per-group risk_coverage curves are degenerate. The real, varying
signal is the per-probe `sufficiency` scalar the curator uses to decide when to stop.
This script ranks predictions by their final-probe sufficiency and sweeps coverage —
"abstain on the least-sufficient answers" — the plan's intended sufficiency-as-abstention.

Only curator policies carry probes; uniform/aks have none (no signal to abstain on) — that
is the point: selective prediction is a capability curation adds. We report, per curator
policy@budget: AURC (lower=better), acc@cov=0.5, and the correlation of sufficiency with
correctness (does higher sufficiency mean more likely right?).

Usage: python scripts/eval_risk_coverage.py --preds-glob 'runs/r1_v2_*_qwen2_5' --budget 8
"""
from __future__ import annotations
import argparse, csv, glob, json, os
from pathlib import Path


def final_sufficiency(rec):
    probes = rec.get("probes") or []
    for p in reversed(probes):
        if isinstance(p, dict) and p.get("sufficiency") is not None:
            return float(p["sufficiency"])
    return None


def risk_coverage(pairs):
    """pairs = [(sufficiency, correct)]; sort desc by sufficiency, sweep coverage."""
    pairs = [p for p in pairs if p[0] is not None]
    if len(pairs) < 3:
        return None, None, None
    pairs.sort(key=lambda x: x[0], reverse=True)
    n = len(pairs)
    curve = []
    for i in range(1, n + 1):
        cov = i / n
        acc = sum(1 for _, ok in pairs[:i] if ok) / i
        curve.append((round(cov, 4), round(acc, 4)))
    # acc at coverage 0.5
    half = curve[max(0, n // 2 - 1)][1]
    # AURC: area under risk(=1-acc) vs coverage
    area = 0.0
    for (c1, a1), (c2, a2) in zip(curve, curve[1:]):
        area += (c2 - c1) * ((1 - a1) + (1 - a2)) / 2.0
    aurc = area / (curve[-1][0] - curve[0][0]) if curve[-1][0] != curve[0][0] else None
    return half, (round(aurc, 4) if aurc is not None else None), curve


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds-glob", required=True)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--policies", nargs="+",
                    default=["scripted_curator", "curator_siglip", "learned_curator"])
    ap.add_argument("--out-csv", default="reports/risk_coverage.csv")
    ap.add_argument("--out-md", default="reports/risk_coverage.md")
    args = ap.parse_args()

    # qa path per dataset stem: runs/r1_v2_<ds>_qwen2_5 -> configs/r1_v2_<ds>_qwen2_5.json.qa_path
    def gold_map(run_dir):
        stem = Path(run_dir).name
        cfg = Path("configs") / f"{stem}.json"
        if not cfg.exists():
            return {}
        qa = json.loads(cfg.read_text()).get("qa_path")
        if not qa or not os.path.exists(qa):
            return {}
        return {q["qid"]: q.get("answer_index") for q in map(json.loads, open(qa))}

    rows = []
    for d in sorted(glob.glob(args.preds_glob)):
        pf = os.path.join(d, "predictions.jsonl")
        if not os.path.exists(pf):
            continue
        ds = Path(d).name
        gold = gold_map(d)
        if not gold:
            print(f"  ! no gold for {ds}, skipping"); continue
        buckets = {p: [] for p in args.policies}
        for line in open(pf):
            r = json.loads(line)
            if r.get("policy") not in buckets or int(r.get("budget", 0)) != args.budget:
                continue
            g = gold.get(r.get("qid"))
            ok = (g is not None and r.get("answer_index") is not None and int(r["answer_index"]) == int(g))
            buckets[r["policy"]].append((final_sufficiency(r), bool(ok)))
        for pol, pairs in buckets.items():
            if not pairs:
                continue
            base_acc = sum(1 for _, ok in pairs if ok) / len(pairs)
            half, aurc, curve = risk_coverage(pairs)
            # correlation sign: mean sufficiency of correct vs incorrect
            suf_ok = [s for s, ok in pairs if ok and s is not None]
            suf_no = [s for s, ok in pairs if not ok and s is not None]
            gap = (sum(suf_ok) / len(suf_ok) - sum(suf_no) / len(suf_no)) if suf_ok and suf_no else None
            rows.append({
                "dataset": ds, "policy": pol, "budget": args.budget, "n": len(pairs),
                "base_acc": round(base_acc, 4),
                "acc_at_cov0.5": half,
                "lift@0.5": round(half - base_acc, 4) if half is not None else None,
                "aurc": aurc,
                "suff_gap(correct-wrong)": round(gap, 4) if gap is not None else None,
            })

    if not rows:
        print("no rows — check policy names / budget"); return 1
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys())); w.writeheader(); w.writerows(rows)
    lines = [f"# E-C Selective prediction via sufficiency (budget={args.budget})", "",
             "Rank answers by final-probe sufficiency, abstain on the least sufficient.",
             "`lift@0.5` = accuracy on the top-50%-sufficient half minus base accuracy (>0 means the",
             "sufficiency signal is useful for abstention). `suff_gap` = mean sufficiency of correct minus",
             "incorrect answers (>0 means higher sufficiency tracks correctness). Baselines (uniform/aks)",
             "have no probes → no abstention signal, which is the point.", "",
             "| dataset | policy | n | base_acc | acc@cov0.5 | lift@0.5 | AURC | suff_gap |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['dataset']} | {r['policy']} | {r['n']} | {r['base_acc']} | "
                     f"{r['acc_at_cov0.5']} | {r['lift@0.5']} | {r['aurc']} | {r['suff_gap(correct-wrong)']} |")
    Path(args.out_md).write_text("\n".join(lines) + "\n")
    print(f"wrote {args.out_csv} ({len(rows)} rows) + {args.out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
