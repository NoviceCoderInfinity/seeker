#!/usr/bin/env python3
"""M2 phase 1: retention-utility caption variants — generation + scoring + GATE.

The one learned lane left open (plan8): train the WRITE turn to produce ledgers that
maximize FUTURE-question answerability. Phase 1 measures the headroom before any
training (the discipline that killed answer-RL): per training video, sample N ledger
variants (1 greedy + N-1 at temperature T), score each by ledger-only answerability of
that video's train questions, and report best-of-N vs greedy.

GATE (pre-registered): if mean(best-of-N) - mean(greedy) < 0.05, the WRITE turn has no
sampling headroom and M2 dies here, before a single training step. If it passes, the
output JSONL is the RFT dataset: best-scoring variant = SFT target for a LoRA on the
caption turn (train on dgx, firewalled videos only — this script only reads traincln).

Usage (inside a job serving Qwen on $PORT):
  python scripts/rft_captions.py --qa data/processed/traincln/nextgqa_traincln.jsonl \
    --frames data/indices/nextgqa_siglip_frames.jsonl \
    --endpoint http://127.0.0.1:8565 --variants 6 --videos-cap 350 \
    --out reports/rft_captions_nextgqa.jsonl
"""
from __future__ import annotations
import argparse, json, random, re
from collections import defaultdict
from pathlib import Path

from vidcurator.io import load_frames, load_examples
from vidcurator.retrieval import FrameIndex
from vidcurator.answerers import build_answerer

SCAN_N = 16
CAPTION_TEMP = 0.9


def uniform_scan(frames, n):
    if len(frames) <= n:
        return list(frames)
    step = len(frames) / n
    return [frames[int(i * step)] for i in range(n)]


def caption_chunk(answerer, chunk, temperature):
    n = len(chunk)
    prompt = "\n".join([
        f"You see {n} video frames in chronological order.",
        f"Write EXACTLY {n} numbered lines. Line i = one factual caption (8-18 words)",
        "of frame i: visible people, objects, actions, text, scene. No preamble.",
        "Format:", "1. <caption of frame 1>",
    ] + ([f"... up to\n{n}. <caption of frame {n}>"] if n > 1 else []))
    payload = {"model": answerer.model,
               "messages": [{"role": "user",
                             "content": answerer._multimodal_content(prompt, chunk)}],
               "temperature": temperature, "max_tokens": 40 * n + 64}
    try:
        raw = answerer._post(payload)
        text = raw["choices"][0]["message"]["content"]
    except Exception:
        return ["" for _ in chunk]
    by_num = {}
    for line in text.splitlines():
        m = re.match(r"^(\d+)[.)]\s*(.+)$", line.strip())
        if m and m.group(2).strip():
            by_num[int(m.group(1))] = m.group(2).strip()
    return [by_num.get(i + 1, "") for i in range(n)]


def build_ledger(answerer, scan, temperature):
    entries = []
    for start in range(0, len(scan), 4):
        chunk = scan[start:start + 4]
        for frame, cap in zip(chunk, caption_chunk(answerer, chunk, temperature)):
            if cap:
                entries.append(f"[t={frame.timestamp:.1f}s scanned] {cap}")
    return entries


def ledger_only_correct(answerer, ex, ledger_text):
    opts = ex.options or []
    ctx = f"Question: {ex.question}"
    if opts:
        ctx += "\nOptions:\n" + "\n".join(
            f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(opts))
    ctx += "\nRunning summary:\n" + ledger_text
    try:
        pred = answerer.answer(ex, [], ctx)
        return int(ex.answer_index is not None and pred.answer_index is not None
                   and int(pred.answer_index) == int(ex.answer_index))
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True, help="traincln QA jsonl (firewalled)")
    ap.add_argument("--frames", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--variants", type=int, default=6)
    ap.add_argument("--videos-cap", type=int, default=350)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    index = FrameIndex(load_frames(args.frames))
    by_video = defaultdict(list)
    for ex in load_examples(args.qa):
        by_video[ex.video_id].append(ex)
    vids = sorted(v for v in by_video if index.frames(v))
    random.Random(0).shuffle(vids)
    # prefer multi-question videos (denser reward), then fill with the rest
    vids = sorted(vids, key=lambda v: -len(by_video[v]))[: args.videos_cap]
    answerer = build_answerer({"kind": "openai_compat", "endpoint": args.endpoint,
                               "send_images": True, "max_images": SCAN_N, "model": "served",
                               "prompt_style": "v3", "timeout_seconds": 600})

    out = Path(args.out)
    done = set()
    if out.exists():  # crash-safe resume
        for line in open(out):
            done.add(json.loads(line)["video_id"])
    fh = open(out, "a")
    greedy_scores, best_scores = [], []
    for i, vid in enumerate(vids):
        if vid in done:
            continue
        scan = uniform_scan(index.frames(vid), SCAN_N)
        qs = by_video[vid]
        variants = []
        for k in range(args.variants):
            temp = 0.0 if k == 0 else CAPTION_TEMP
            entries = build_ledger(answerer, scan, temp)
            ledger = "\n".join(entries)
            score = (sum(ledger_only_correct(answerer, ex, ledger) for ex in qs) / len(qs)
                     if entries else 0.0)
            variants.append({"variant": k, "temperature": temp,
                             "entries": entries, "score": score})
        greedy_scores.append(variants[0]["score"])
        best_scores.append(max(v["score"] for v in variants))
        fh.write(json.dumps({"video_id": vid, "n_questions": len(qs),
                             "scan_frame_ids": [f.frame_id for f in scan],
                             "variants": variants}) + "\n")
        fh.flush()
        if (i + 1) % 20 == 0:
            g = sum(greedy_scores) / len(greedy_scores)
            b = sum(best_scores) / len(best_scores)
            print(f"{i+1}/{len(vids)} videos | greedy={g:.3f} best-of-{args.variants}={b:.3f} "
                  f"headroom={b-g:+.3f}", flush=True)
    fh.close()
    if greedy_scores:
        g = sum(greedy_scores) / len(greedy_scores)
        b = sum(best_scores) / len(best_scores)
        verdict = "PASS" if b - g >= 0.05 else "FAIL"
        print(f"FINAL n={len(greedy_scores)} greedy={g:.3f} best={b:.3f} "
              f"headroom={b-g:+.3f} GATE={verdict}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
