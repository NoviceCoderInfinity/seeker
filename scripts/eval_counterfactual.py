#!/usr/bin/env python3
"""E-I: Counterfactual Evidence Sufficiency (CES) — causal test of evidence use.

For every CORRECT answer whose selected frames include >=1 gold-evidence frame, re-ask the
VLM under two interventions and measure the flip rate (correct -> wrong):

  remove  : answer from the selected frames MINUS the evidence frames.
  replace : answer from the selected non-evidence frames PLUS same-count distractor frames
            sampled from OUTSIDE the gold span.

flip_rate = fraction that flip to wrong. HIGH flip = the answer was causally grounded in the
evidence (good). LOW flip = the model didn't need the evidence — right answer, wrong reason.
"N% of <policy>'s correct answers survive evidence removal" = N% ungrounded.

Runs against a live OpenAI-compatible VLM endpoint (serve it first, e.g. jobs/12 pattern).
NExT-GQA only (needs gold evidence_spans). Use --limit for a smoke test before the full run.

Usage (inside a job that serves the VLM on $PORT):
  python scripts/eval_counterfactual.py \
    --preds runs/r1_v2_nextgqa_qwen2_5/predictions.jsonl \
    --qa data/processed/nextgqa_val_150.jsonl \
    --frames data/indices/nextgqa_siglip_frames.jsonl \
    --endpoint http://127.0.0.1:8080 --budget 8 \
    --policies uniform aks scripted_curator curator_siglip \
    --out reports/ei_counterfactual_nextgqa.md [--limit 5]
"""
from __future__ import annotations
import argparse, json, random
from collections import defaultdict
from pathlib import Path

from vidcurator.io import load_frames, load_examples
from vidcurator.retrieval import FrameIndex
from vidcurator.answerers import build_answerer


def in_span(ts, spans, tol=1.0):
    return any(s.start - tol <= ts <= s.end + tol for s in spans)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--budget", type=int, default=8)
    ap.add_argument("--policies", nargs="+",
                    default=["uniform", "aks", "scripted_curator", "curator_siglip"])
    ap.add_argument("--max-images", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: cap correct-cases per policy")
    ap.add_argument("--out", default="reports/ei_counterfactual_nextgqa.md")
    args = ap.parse_args()
    random.seed(0)

    index = FrameIndex(load_frames(args.frames))
    examples = {e.qid: e for e in load_examples(args.qa)}
    answerer = build_answerer({
        "kind": "openai_compat", "endpoint": args.endpoint, "send_images": True,
        "max_images": args.max_images, "model": "served",
    })

    def is_ok(pred_idx, ex):
        return ex.answer_index is not None and pred_idx is not None and int(pred_idx) == int(ex.answer_index)

    def reanswer(ex, frames):
        try:
            pred = answerer.answer(ex, frames, "")
            return pred.answer_index
        except Exception as e:  # keep the sweep alive; count as non-flip (conservative)
            print(f"  ! answer failed qid={ex.qid}: {e}", flush=True)
            return None

    stats = defaultdict(lambda: {"eligible": 0, "remove_flip": 0, "replace_flip": 0})
    counts = defaultdict(int)
    for line in open(args.preds):
        r = json.loads(line)
        pol, bud = r.get("policy"), int(r.get("budget", 0))
        if pol not in args.policies or bud != args.budget:
            continue
        ex = examples.get(r.get("qid"))
        if not ex or not ex.evidence_spans:
            continue
        if not is_ok(r.get("answer_index"), ex):
            continue  # only test correct answers
        if args.limit and counts[pol] >= args.limit:
            continue
        # reconstruct selected FrameRecords
        sel = [index.get(ex.video_id, fid) for fid in (r.get("selected_frame_ids") or [])]
        sel = [f for f in sel if f is not None]
        if not sel:
            continue
        ev = [f for f in sel if in_span(f.timestamp, ex.evidence_spans)]
        non_ev = [f for f in sel if not in_span(f.timestamp, ex.evidence_spans)]
        if not ev:
            continue  # no evidence frame in the selection -> nothing to ablate
        counts[pol] += 1
        st = stats[pol]
        st["eligible"] += 1
        # (a) remove evidence frames — one call, flip only counts a definite wrong answer
        a = reanswer(ex, non_ev)
        if a is not None and not is_ok(a, ex):
            st["remove_flip"] += 1
        # (b) replace with distractors from outside the span — one call
        pool = [f for f in index.frames(ex.video_id)
                if not in_span(f.timestamp, ex.evidence_spans) and f.frame_id not in {s.frame_id for s in sel}]
        distractors = random.sample(pool, min(len(ev), len(pool))) if pool else []
        b = reanswer(ex, non_ev + distractors)
        if b is not None and not is_ok(b, ex):
            st["replace_flip"] += 1

    lines = [f"# E-I Counterfactual Evidence Sufficiency — NExT-GQA (budget={args.budget})", "",
             "Higher flip rate = answers are causally grounded in the evidence frames.",
             "`survive_remove` = correct answers that stay correct after evidence removal = ungrounded.", "",
             "| policy | eligible | remove_flip | replace_flip | survive_remove |",
             "|---|---|---|---|---|"]
    for pol in args.policies:
        s = stats[pol]
        e = s["eligible"] or 1
        lines.append(f"| {pol} | {s['eligible']} | {s['remove_flip']} ({100*s['remove_flip']/e:.0f}%) | "
                     f"{s['replace_flip']} ({100*s['replace_flip']/e:.0f}%) | "
                     f"{e - s['remove_flip']} ({100*(e-s['remove_flip'])/e:.0f}%) |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
