#!/usr/bin/env python3
"""G2: temporal-window oracle on hour-long video (annotation-free headroom gate).

LVBench/Video-MME-long have no gold spans, so the selection-headroom ceiling is
measured directly: split each video's frame index into W contiguous windows, answer
the question independently from 8 uniform frames of EACH window, and report
  - per-window accuracy (does any fixed temporal prior beat uniform?),
  - ANY-window oracle (ceiling for any selector/memory that picks one window),
  - margin over the full-video uniform@8 baseline.
If the any-window oracle sits near the uniform baseline + noise, no memory design
can rescue the paradigm at this answerer scale and the long-video wall is a
capability wall — measured, not asserted. If the oracle is high, a learned
window-scorer has a real target and plan8 M3 proceeds.

Usage (inside a job serving Qwen on $PORT):
  python scripts/window_oracle.py --qa data/processed/heldout/lvbench_heldout.jsonl \
    --frames data/indices/lvbench_siglip_frames.jsonl \
    --endpoint http://127.0.0.1:8563 --windows 8 --per-window 8 \
    --out reports/window_oracle_lvbench.md
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from vidcurator.io import load_frames, load_examples
from vidcurator.retrieval import FrameIndex
from vidcurator.answerers import build_answerer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--per-window", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index = FrameIndex(load_frames(args.frames))
    examples = load_examples(args.qa)
    if args.limit:
        examples = examples[: args.limit]
    answerer = build_answerer({"kind": "openai_compat", "endpoint": args.endpoint,
                               "send_images": True, "max_images": args.per_window,
                               "model": "served", "prompt_style": "v3",
                               "timeout_seconds": 600})

    W = args.windows
    per_q = []
    for ex in examples:
        frames = index.frames(ex.video_id)
        if not frames:
            continue
        n = len(frames)
        row = {"qid": ex.qid, "gold": ex.answer_index, "windows": []}
        for w in range(W):
            chunk = frames[w * n // W:(w + 1) * n // W]
            if not chunk:
                row["windows"].append(None)
                continue
            step = max(1, len(chunk) // args.per_window)
            picked = chunk[::step][: args.per_window]
            opts = ex.options or []
            ctx = f"Question: {ex.question}"
            if opts:
                ctx += "\nOptions:\n" + "\n".join(
                    f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(opts))
            try:
                pred = answerer.answer(ex, picked, ctx)
                row["windows"].append(pred.answer_index)
            except Exception as e:
                print(f"  ! qid={ex.qid} w={w}: {e}", flush=True)
                row["windows"].append(None)
        per_q.append(row)
        if len(per_q) % 25 == 0:
            print(f"{len(per_q)} questions done", flush=True)

    n = len(per_q)
    win_acc = []
    for w in range(W):
        ok = sum(1 for r in per_q if r["windows"][w] is not None
                 and int(r["windows"][w]) == r["gold"])
        win_acc.append(ok / n if n else 0.0)
    any_ok = sum(1 for r in per_q if any(
        a is not None and int(a) == r["gold"] for a in r["windows"])) / max(n, 1)
    maj = 0
    for r in per_q:
        votes = [a for a in r["windows"] if a is not None]
        if votes:
            top = max(set(votes), key=votes.count)
            maj += int(top == r["gold"])
    lines = [f"# Temporal-window oracle — {Path(args.qa).stem} (W={W}, {args.per_window} frames/window, n={n})",
             "", "| window | acc |", "|---|---|"]
    lines += [f"| w{w} | {a:.3f} |" for w, a in enumerate(win_acc)]
    lines += ["", f"- best fixed window: {max(win_acc):.3f}",
              f"- ANY-window oracle: {any_ok:.3f}",
              f"- window majority vote: {maj / max(n, 1):.3f}"]
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".per_question.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_q) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
