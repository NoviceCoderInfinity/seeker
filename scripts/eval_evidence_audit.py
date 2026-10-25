#!/usr/bin/env python3
"""E-A: Evidence-faithfulness audit for NExT-GQA (gold temporal spans available).

For every policy, joins predictions with gold evidence spans and reports, per budget:
  - FEP  (frame-evidence precision): fraction of selected frames inside a gold span
  - FER  (frame-evidence recall):    fraction of gold spans hit by >=1 selected frame
  - acc:  accuracy
  - pct_correct_zero_evidence: of CORRECT answers, the fraction that selected ZERO
    gold-evidence frames  ==  "right answer, wrong frames" (answered from prior, not evidence)

The last column is the headline: a high value means the method's correct answers are not
grounded in the frames it chose. Pure post-hoc; no GPU.

Usage: python scripts/eval_evidence_audit.py \
    --preds runs/r1_v2_nextgqa_qwen2_5/predictions.jsonl \
    --qa data/processed/nextgqa_val_150.jsonl [--budget 8]
"""
from __future__ import annotations
import argparse
import json
from collections import defaultdict
from pathlib import Path


def load_gold(qa_path: str) -> dict[str, dict]:
    gold = {}
    for line in open(qa_path):
        q = json.loads(line)
        spans = [(float(s["start"]), float(s["end"])) for s in q.get("evidence_spans", []) or []]
        gold[q["qid"]] = {"answer_index": q.get("answer_index"), "spans": spans}
    return gold


def ts_in_spans(ts: float, spans: list[tuple[float, float]], tol: float = 0.5) -> bool:
    return any(s - tol <= ts <= e + tol for s, e in spans)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--budget", type=int, default=None, help="filter to one budget; default = all")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    gold = load_gold(args.qa)
    # per (policy, budget): accumulators
    agg: dict[tuple[str, int], dict] = defaultdict(lambda: {
        "n": 0, "correct": 0, "fep_sum": 0.0, "fer_sum": 0.0,
        "correct_zero_ev": 0, "n_with_spans": 0})

    for line in open(args.preds):
        r = json.loads(line)
        qid = r.get("qid")
        g = gold.get(qid)
        if g is None or not g["spans"]:
            continue  # only questions with gold evidence spans count for FEP/FER
        pol, bud = r.get("policy"), int(r.get("budget", 0))
        if args.budget is not None and bud != args.budget:
            continue
        ts = [float(t) for t in (r.get("selected_timestamps") or [])]
        spans = g["spans"]
        in_ev = [t for t in ts if ts_in_spans(t, spans)]
        fep = (len(in_ev) / len(ts)) if ts else 0.0
        fer = (sum(1 for s, e in spans if any(s - 0.5 <= t <= e + 0.5 for t in ts)) / len(spans))
        correct = (r.get("answer_index") == g["answer_index"])

        k = (pol, bud)
        a = agg[k]
        a["n"] += 1
        a["n_with_spans"] += 1
        a["fep_sum"] += fep
        a["fer_sum"] += fer
        if correct:
            a["correct"] += 1
            if not in_ev:
                a["correct_zero_ev"] += 1

    rows = []
    for (pol, bud), a in sorted(agg.items()):
        n = a["n"] or 1
        c = a["correct"] or 1
        rows.append({
            "policy": pol, "budget": bud, "n": a["n"],
            "acc": a["correct"] / n,
            "fep": a["fep_sum"] / n,
            "fer": a["fer_sum"] / n,
            "pct_correct_zero_evidence": a["correct_zero_ev"] / c,
        })

    # print table
    hdr = f"{'policy':<22}{'bud':>4}{'n':>6}{'acc':>8}{'FEP':>8}{'FER':>8}{'%correct_no_ev':>16}"
    print(hdr)
    print("-" * len(hdr))
    for r in rows:
        print(f"{r['policy']:<22}{r['budget']:>4}{r['n']:>6}{r['acc']:>8.3f}"
              f"{r['fep']:>8.3f}{r['fer']:>8.3f}{r['pct_correct_zero_evidence']:>16.3f}")

    if args.out:
        Path(args.out).write_text(json.dumps(rows, indent=2))
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
