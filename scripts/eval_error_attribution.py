#!/usr/bin/env python3
"""E-N: error attribution for curator policies on NExT-GQA (needs gold evidence spans).

For every WRONG answer, attribute the failure to one of four causes using the saved
trace + gold spans:

  reasoning-fail  : a gold-evidence frame WAS in the final buffer, model still wrong.
  curation-drop   : gold evidence was retrieved as a candidate but dropped/compressed
                    out of the final buffer.
  stop-too-early  : the loop answered without ever retrieving (0 retrievals) and the
                    seed frames missed the evidence — sufficiency fired too soon.
  retrieval-miss  : the loop did retrieve, but the gold moment was never among the
                    retrieved candidates (retriever/pool ceiling).

"The buffer makes failure attributable" — a black-box selector cannot produce this table.

Usage: python scripts/eval_error_attribution.py --run runs/r1_v2_nextgqa_qwen2_5 \
         --qa data/processed/nextgqa_val_150.jsonl \
         --policies curator_siglip scripted_curator --out reports/error_attribution.md
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path


def load_gold(qa_path):
    gold = {}
    for line in open(qa_path):
        q = json.loads(line)
        spans = [(float(s["start"]), float(s["end"])) for s in (q.get("evidence_spans") or [])]
        gold[q["qid"]] = {"ai": q.get("answer_index"), "spans": spans}
    return gold


def in_spans(ts, spans, tol=1.0):
    return any(s - tol <= ts <= e + tol for s, e in spans)


def fid_to_ts(fid):
    try:
        return float(int(fid))  # 1 fps: frame_id NNNNNN == second
    except (ValueError, TypeError):
        return None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--policies", nargs="+", default=["curator_siglip", "scripted_curator"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[2, 4, 8, 16, 32])
    ap.add_argument("--out", default="reports/error_attribution.md")
    args = ap.parse_args()

    gold = load_gold(args.qa)
    preds = Path(args.run) / "predictions.jsonl"
    # (policy, budget) -> Counter of buckets; also correct/total
    stats = defaultdict(lambda: {"reasoning-fail": 0, "curation-drop": 0,
                                 "stop-too-early": 0, "retrieval-miss": 0,
                                 "no-gold": 0, "correct": 0, "wrong": 0})
    for line in open(preds):
        r = json.loads(line)
        pol, bud = r.get("policy"), int(r.get("budget", 0))
        if pol not in args.policies or bud not in args.budgets:
            continue
        g = gold.get(r.get("qid"))
        if not g:
            continue
        st = stats[(pol, bud)]
        ok = (g["ai"] is not None and r.get("answer_index") is not None
              and int(r["answer_index"]) == int(g["ai"]))
        if ok:
            st["correct"] += 1
            continue
        st["wrong"] += 1
        spans = g["spans"]
        if not spans:
            st["no-gold"] += 1
            continue
        final_ts = [float(t) for t in (r.get("selected_timestamps") or [])]
        evidence_in_final = any(in_spans(t, spans) for t in final_ts)
        retrieved_ts = []
        for e in r.get("trace", []):
            if e.get("event") == "retrieve":
                for fr in e.get("frames", []):
                    t = fid_to_ts(fr.get("frame_id"))
                    if t is not None:
                        retrieved_ts.append(t)
        evidence_retrieved = any(in_spans(t, spans) for t in retrieved_ts)
        n_ret = int(r.get("num_retrievals", 0) or 0)
        if evidence_in_final:
            st["reasoning-fail"] += 1
        elif evidence_retrieved:
            st["curation-drop"] += 1
        elif n_ret == 0:
            st["stop-too-early"] += 1
        else:
            st["retrieval-miss"] += 1

    lines = ["# E-N Error attribution — NExT-GQA (curator policies)", "",
             "Distribution of *wrong* answers by cause. reasoning-fail = had evidence, still wrong;",
             "curation-drop = retrieved evidence then dropped it; stop-too-early = answered without",
             "retrieving; retrieval-miss = evidence never retrieved.", "",
             "| policy | K | wrong | reasoning-fail | curation-drop | stop-too-early | retrieval-miss |",
             "|---|---|---|---|---|---|---|"]
    for (pol, bud) in sorted(stats):
        s = stats[(pol, bud)]
        w = s["wrong"] or 1
        def pct(k): return f"{s[k]} ({100*s[k]/w:.0f}%)"
        lines.append(f"| {pol} | {bud} | {s['wrong']} | {pct('reasoning-fail')} | "
                     f"{pct('curation-drop')} | {pct('stop-too-early')} | {pct('retrieval-miss')} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
