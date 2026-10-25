#!/usr/bin/env python3
"""Answerer-side pass@k probe: is the ANSWER turn the capped stage, or does it have headroom?

Replays the exact stored buffers of a completed run (selected_frame_ids + metadata.running_summary
+ last probe, i.e. everything buffer.render_text() contained) and asks the served answerer
k times at --temperature, plus once greedy as the replay-fidelity control.

Decomposition logic (companion to scripts/passk_score.py, which samples the CURATOR):
  curator-side pass@32 ~= pass@1 (measured: +0.010)  AND  answerer-side pass@32 >> pass@1
    => the headroom lives in the answer turn (answer-RL / self-consistency territory);
  both flat => the pipeline is genuinely capped at this evidence level; only better
    evidence (ledger streams, subtitles) or a stronger answerer moves the number.

Usage (inside a job serving the answerer with sampling support on $PORT):
  python scripts/answerer_passk.py \
      --preds runs/p0_v3control_nextgqa/_shards/b8_dde_curator.jsonl \
      --qa data/processed/heldout/nextgqa_heldout_p120.jsonl \
      --frames data/indices/nextgqa_siglip_frames.jsonl \
      --endpoint http://127.0.0.1:8485 --k 32 --temperature 1.0 \
      --out runs/answerer_passk32_nextgqa_b8/votes.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os

from vidcurator.answerers import build_answerer
from vidcurator.io import load_examples, load_frames
from vidcurator.retrieval import FrameIndex


def unbiased_pass_at_j(k: int, c: int, j: int) -> float:
    if c == 0:
        return 0.0
    if k - c < j:
        return 1.0
    return 1.0 - math.comb(k - c, j) / math.comb(k, j)


def rebuild_context(row: dict, ex, frames) -> str:
    """Reconstruct buffer.render_text() from the stored prediction row."""
    lines = [f"Question: {ex.question}"]
    if ex.options:
        lines.append("Options:")
        for idx, option in enumerate(ex.options):
            letter = chr(ord("A") + idx) if idx < 26 else str(idx)
            lines.append(f"{letter}. {option}")
    summary = (row.get("metadata") or {}).get("running_summary") or ""
    if summary:
        lines.append("Running summary:")
        lines.append(summary)
    lines.append("Keyframes:")
    for fr in sorted(frames, key=lambda f: f.timestamp):
        # usefulness scores are not persisted; 0.500 placeholder (greedy control quantifies this)
        lines.append(f"- id={fr.frame_id} t={fr.timestamp:.1f}s use=0.500: {fr.caption or ''}")
    probes = row.get("probes") or []
    if probes:
        last = probes[-1]
        lines.append(
            f"Last probe: sufficiency={float(last.get('sufficiency', 0.0)):.2f}; "
            f"missing={last.get('missing_moment', '')}; next={last.get('next_action', '')}"
        )
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds", required=True, help="shard jsonl of the run to replay")
    ap.add_argument("--qa", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--k", type=int, default=32)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--max-images", type=int, default=16)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index = FrameIndex(load_frames(args.frames))
    examples = {e.qid: e for e in load_examples(args.qa)}

    base = dict(kind="openai_compat", endpoint=args.endpoint, model="served",
                send_images=True, max_images=args.max_images, timeout_seconds=600,
                prompt_style="v3")
    greedy = build_answerer(base)
    sampler = build_answerer({**base, "temperature": args.temperature})

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if os.path.exists(args.out):
        for line in open(args.out):
            done.add(json.loads(line)["qid"])

    per_q = []
    with open(args.out, "a") as fh:
        for line in open(args.preds):
            row = json.loads(line)
            ex = examples.get(row["qid"])
            if ex is None or row["qid"] in done:
                continue
            frames = [index.get(ex.video_id, fid) for fid in (row.get("selected_frame_ids") or [])]
            frames = [f for f in frames if f is not None]
            if not frames:
                continue
            ctx = rebuild_context(row, ex, frames)
            try:
                g = greedy.answer(ex, frames, ctx)
                votes = [sampler.answer(ex, frames, ctx).answer_index for _ in range(args.k)]
            except Exception as e:
                print(f"  ! failed qid={ex.qid}: {e}", flush=True)
                continue
            rec = {"qid": ex.qid, "gold": ex.answer_index, "orig": row.get("answer_index"),
                   "greedy_replay": g.answer_index, "votes": votes}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            print(f"  {ex.qid}: orig={rec['orig']} replay={rec['greedy_replay']} "
                  f"correct_votes={sum(1 for v in votes if v is not None and int(v) == ex.answer_index)}/{args.k}",
                  flush=True)

    # score everything on file (including prior partial runs)
    for line in open(args.out):
        r = json.loads(line)
        gold = r["gold"]
        c = sum(1 for v in r["votes"] if v is not None and int(v) == gold)
        per_q.append((len(r["votes"]), c,
                      r["orig"] is not None and int(r["orig"]) == gold,
                      r["greedy_replay"] is not None and int(r["greedy_replay"]) == gold))
    if not per_q:
        raise SystemExit("no scored rows")
    n = len(per_q)
    kmax = min(k for k, _, _, _ in per_q)
    print(f"\n== answerer pass@k  n={n}  k={kmax}  T={args.temperature} ==")
    print(f"  orig-run acc      = {sum(1 for _, _, o, _ in per_q if o) / n:.3f}")
    print(f"  greedy replay acc = {sum(1 for _, _, _, g in per_q if g) / n:.3f}  (fidelity control)")
    for j in (1, 2, 4, 8, 16, 32):
        if j > kmax:
            continue
        p = sum(unbiased_pass_at_j(k, c, j) for k, c, _, _ in per_q) / n
        print(f"  pass@{j:<3d} = {p:.3f}")
    print(f"  questions with >=1 correct sample = {sum(1 for _, c, _, _ in per_q if c > 0) / n:.3f}")


if __name__ == "__main__":
    main()
