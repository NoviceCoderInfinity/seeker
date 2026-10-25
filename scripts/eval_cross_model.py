#!/usr/bin/env python3
"""E-G: cross-model curation — is the curated buffer a portable artifact?

The curator's buffer (its selected frames) is already saved per prediction. E-G re-answers
those SAME frames with a DIFFERENT (stronger) answerer served at --endpoint, e.g. a Qwen2.5-7B
curator's 8-frame buffer answered by Qwen3-VL. Because the buffer is saved we don't need to
dual-serve live: serve only the answerer and replay the frame sets.

Deliverable per policy@budget: accuracy of <answerer> on <policy>'s selected frames. Compare
to the same answerer's uniform@K (from the R5 run) and to the original curator accuracy — if
answerer-on-curated-buffer > answerer-on-uniform, curation transfers across models.

Usage (inside a job serving the cross-answerer, e.g. Qwen3, on $PORT):
  python scripts/eval_cross_model.py \
    --preds runs/r1_v2_nextgqa_qwen2_5/predictions.jsonl \
    --qa data/processed/nextgqa_val_150.jsonl \
    --frames data/indices/nextgqa_siglip_frames.jsonl \
    --endpoint http://127.0.0.1:8080 \
    --policies uniform curator_siglip --budgets 8 \
    --out reports/eg_cross_model_nextgqa.md [--limit N]
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

from vidcurator.io import load_frames, load_examples
from vidcurator.retrieval import FrameIndex
from vidcurator.answerers import build_answerer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True)
    ap.add_argument("--qa", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--policies", nargs="+", default=["uniform", "curator_siglip"])
    ap.add_argument("--budgets", nargs="+", type=int, default=[8])
    ap.add_argument("--max-images", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--prompt-style", default="v3")
    ap.add_argument("--with-ledger", action="store_true",
                    help="carry the stored running summary + keyframe captions + last probe "
                         "into the replay context (ledger portability, L4) instead of the "
                         "minimal question+options context")
    ap.add_argument("--ledger-only", action="store_true",
                    help="L1: send the ledger context but ZERO images — measures how much "
                         "of the answer the text memory alone carries (implies --with-ledger)")
    ap.add_argument("--strip-timestamps", action="store_true",
                    help="L3: remove t=..s timestamps from the ledger context — tests "
                         "whether the answerer uses 'when', not just 'what'")
    ap.add_argument("--episodic", choices=["off", "pred", "gold"], default="off",
                    help="M4: append previously answered questions about the SAME video "
                         "to the context. 'pred' uses the stored policy answers (realized "
                         "cache), 'gold' uses ground truth (ceiling).")
    ap.add_argument("--out", default="reports/eg_cross_model.md")
    args = ap.parse_args()
    if args.ledger_only:
        args.with_ledger = True

    index = FrameIndex(load_frames(args.frames))
    examples = {e.qid: e for e in load_examples(args.qa)}
    answerer = build_answerer({"kind": "openai_compat", "endpoint": args.endpoint,
                               "send_images": True, "max_images": args.max_images, "model": "served",
                               "prompt_style": args.prompt_style, "timeout_seconds": 600})

    stats = defaultdict(lambda: {"n": 0, "orig_correct": 0, "xmodel_correct": 0})
    counts = defaultdict(int)
    per_question = []  # sidecar for paired significance tests

    # M4 episodic cache: per (policy, budget, video) -> qid-ordered prior Q&A entries.
    episodes = defaultdict(list)
    if args.episodic != "off":
        rows = sorted((json.loads(l) for l in open(args.preds)), key=lambda r: r.get("qid") or "")
        for r in rows:
            pol, bud = r.get("policy"), int(r.get("budget", 0))
            ex = examples.get(r.get("qid"))
            if pol not in args.policies or bud not in args.budgets or not ex:
                continue
            ans = ex.answer if args.episodic == "gold" else r.get("answer")
            if ans:
                episodes[(pol, bud, ex.video_id)].append((ex.qid, ex.question, str(ans)))
    for line in open(args.preds):
        r = json.loads(line)
        pol, bud = r.get("policy"), int(r.get("budget", 0))
        if pol not in args.policies or bud not in args.budgets:
            continue
        ex = examples.get(r.get("qid"))
        if not ex:
            continue
        key = (pol, bud)
        if args.limit and counts[key] >= args.limit:
            continue
        frames = [index.get(ex.video_id, fid) for fid in (r.get("selected_frame_ids") or [])]
        frames = [f for f in frames if f is not None]
        if not frames:
            continue
        counts[key] += 1
        st = stats[key]
        st["n"] += 1
        # original curator answer correctness (from saved prediction)
        if ex.answer_index is not None and r.get("answer_index") is not None \
                and int(r["answer_index"]) == int(ex.answer_index):
            st["orig_correct"] += 1
        # cross-model re-answer.
        # NOTE: the answerer prompt gets the question + options ONLY via rendered_context
        # (see OpenAICompatAnswerer._prompt / buffer.render_text). Passing "" here makes the
        # model answer with no question/options -> ~random. Reconstruct a minimal context.
        opts = ex.options or []
        ctx = f"Question: {ex.question}"
        if opts:
            ctx += "\nOptions:\n" + "\n".join(
                f"{chr(ord('A') + i) if i < 26 else i}. {o}" for i, o in enumerate(opts)
            )
        if args.with_ledger:
            summary = (r.get("metadata") or {}).get("running_summary") or ""
            if summary:
                ctx += "\nRunning summary:\n" + summary
            ctx += "\nKeyframes:"
            for fr in sorted(frames, key=lambda f: f.timestamp):
                ctx += f"\n- id={fr.frame_id} t={fr.timestamp:.1f}s use=0.500: {fr.caption or ''}"
            probes = r.get("probes") or []
            if probes:
                last = probes[-1]
                ctx += (f"\nLast probe: sufficiency={float(last.get('sufficiency', 0.0)):.2f}; "
                        f"missing={last.get('missing_moment', '')}; next={last.get('next_action', '')}")
        if args.episodic != "off":
            # streaming order: only questions strictly BEFORE this one (qid order) may be
            # in the cache — same-video paraphrases would otherwise leak gold answers.
            prior = [(q, a) for qid2, q, a in episodes.get((pol, bud, ex.video_id), [])
                     if qid2 < ex.qid][:8]
            if prior:
                ctx += "\nPreviously answered questions about this same video:"
                for q, a in prior:
                    ctx += f"\n- Q: {q} A: {a}"
        if args.strip_timestamps:
            ctx = re.sub(r"\[?t=\d+(?:\.\d+)?s(?: scanned)?\]?:?\s*", "", ctx)
        try:
            pred = answerer.answer(ex, [] if args.ledger_only else frames, ctx)
            x_ok = ex.answer_index is not None and pred.answer_index is not None \
                and int(pred.answer_index) == int(ex.answer_index)
            if x_ok:
                st["xmodel_correct"] += 1
            per_question.append({"qid": ex.qid, "policy": pol, "budget": bud,
                                 "xmodel_answer_index": pred.answer_index,
                                 "xmodel_correct": bool(x_ok)})
        except Exception as e:
            print(f"  ! xmodel answer failed qid={ex.qid}: {e}", flush=True)

    lines = ["# E-G Cross-model curation — buffer portability", "",
             "`orig_acc` = original answerer on these frames; `xmodel_acc` = cross-answerer",
             "(served endpoint) on the SAME frames. xmodel_acc > (its own uniform@K) ⇒ the",
             "curated buffer transfers across models.", "",
             "| policy | K | n | orig_acc | xmodel_acc |", "|---|---|---|---|---|"]
    for (pol, bud) in sorted(stats):
        s = stats[(pol, bud)]
        n = s["n"] or 1
        lines.append(f"| {pol} | {bud} | {s['n']} | {s['orig_correct']/n:.4f} | {s['xmodel_correct']/n:.4f} |")
    Path(args.out).write_text("\n".join(lines) + "\n")
    pq_path = Path(args.out).with_suffix(".per_question.jsonl")
    pq_path.write_text("\n".join(json.dumps(r) for r in per_question) + "\n")
    print("\n".join(lines))
    print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
