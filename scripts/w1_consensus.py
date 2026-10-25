#!/usr/bin/env python3
"""W1/W2: consensus WRITE and reliability-tagged memory (plan8-B).

M2 died on the chance-null: picking the best-scoring caption variant per video is
max-of-noise, not headroom (real excess -0.029 nextgqa / +0.036 ego, both below the
0.05 gate). Consensus is the label-free alternative that is *immune* to that trap: it
never looks at the eval signal. For each timestamp slot we keep the medoid caption
across the sampled variants -- the one with the highest mean token overlap to its
siblings -- which suppresses per-sample hallucination without any training.

W2 adds reliability: when the variants disagree about a slot (low mean overlap), the
ledger says so with an [unstable] tag, letting the answerer discount that line.

READ is held fixed across arms: every arm answers from the *same* retrieved frames,
so any difference is attributable to the ledger text alone.

Arms: greedy (variant 0, the control), consensus (W1), tagged (W2).

Usage (inside a job serving Qwen on $PORT):
  python scripts/w1_consensus.py --variants reports/w1_variants_nextgqa_heldout.jsonl \
    --qa data/processed/heldout/nextgqa_heldout.jsonl \
    --frames data/indices/nextgqa_siglip_frames.jsonl \
    --endpoint http://127.0.0.1:8567 --k 8 --out reports/w1_consensus_nextgqa.jsonl
"""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

from vidcurator.io import load_frames, load_examples
from vidcurator.retrieval import (FrameIndex, MMRRetriever, query_embedding_for,
                                  token_set)
from vidcurator.answerers import build_answerer

SLOT_RE = re.compile(r"^\[t=(\d+(?:\.\d+)?)s scanned\]\s*(.*)$")
# W2 tags the least-reliable quartile of slots. A fixed cutoff is wrong here: mean
# slot agreement is ~0.24, so any absolute threshold near it tags half the ledger and
# the tag stops carrying information. The cutoff is set per corpus, label-free.
UNSTABLE_PERCENTILE = 25


def parse_slots(entries):
    """entry string -> (timestamp, caption); tolerate dropped/empty lines."""
    out = []
    for e in entries:
        m = SLOT_RE.match(e)
        if m and m.group(2).strip():
            out.append((float(m.group(1)), m.group(2).strip()))
    return out


def jaccard(a, b):
    if not a or not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def consensus_slots(variants):
    """Medoid caption per timestamp slot across variants -> [(ts, caption, agreement)]."""
    by_ts = defaultdict(list)
    for v in variants:
        for ts, cap in parse_slots(v["entries"]):
            by_ts[ts].append(cap)
    slots = []
    for ts in sorted(by_ts):
        caps = by_ts[ts]
        toks = [token_set(c) for c in caps]
        # medoid = caption with the highest mean overlap to its siblings
        best_i, best_mean = 0, -1.0
        for i in range(len(caps)):
            others = [jaccard(toks[i], toks[j]) for j in range(len(caps)) if j != i]
            mean = sum(others) / len(others) if others else 1.0
            if mean > best_mean:
                best_i, best_mean = i, mean
        slots.append((ts, caps[best_i], best_mean))
    return slots


def render(slots, cutoff=None):
    """Render slots as ledger text; tag slots below ``cutoff`` as [unstable]."""
    lines = []
    for ts, cap, agree in slots:
        mark = " [unstable]" if (cutoff is not None and agree < cutoff) else ""
        lines.append(f"[t={ts:.1f}s scanned]{mark} {cap}")
    return "\n".join(lines)


def greedy_ledger(variants):
    v0 = next((v for v in variants if v["variant"] == 0), variants[0])
    return "\n".join(v0["entries"])


def answer_with(answerer, ex, frames, ledger_text, tagged: bool):
    opts = ex.options or []
    ctx = f"Question: {ex.question}"
    if opts:
        ctx += "\nOptions:\n" + "\n".join(
            f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(opts))
    ctx += "\nRunning summary:\n" + ledger_text
    if tagged:
        ctx += ("\n(Lines marked [unstable] were described inconsistently across "
                "passes; trust them less than unmarked lines.)")
    try:
        pred = answerer.answer(ex, frames, ctx)
        return int(ex.answer_index is not None and pred.answer_index is not None
                   and int(pred.answer_index) == int(ex.answer_index))
    except Exception:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variants", required=True, help="w1_variants_*.jsonl")
    ap.add_argument("--qa", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--k", type=int, default=8, help="retrieved frames (READ, fixed)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    # pass 1: build medoid slots and pool agreements to fix the W2 cutoff corpus-wide
    slots_by_video, greedy_by_video, pool = {}, {}, []
    for line in open(args.variants):
        r = json.loads(line)
        slots_by_video[r["video_id"]] = consensus_slots(r["variants"])
        greedy_by_video[r["video_id"]] = greedy_ledger(r["variants"])
        pool.extend(a for _, _, a in slots_by_video[r["video_id"]])
    pool.sort()
    cutoff = pool[max(0, int(len(pool) * UNSTABLE_PERCENTILE / 100) - 1)] if pool else 0.0
    print(f"W2 unstable cutoff (p{UNSTABLE_PERCENTILE}) = {cutoff:.3f}", flush=True)

    ledgers = {}
    for vid, slots in slots_by_video.items():
        agrees = [a for _, _, a in slots]
        ledgers[vid] = {"greedy": greedy_by_video[vid],
                        "consensus": render(slots),
                        "tagged": render(slots, cutoff=cutoff),
                        "agreement": sum(agrees) / len(agrees) if agrees else 0.0}

    index = FrameIndex(load_frames(args.frames))
    retriever = MMRRetriever(index)
    answerer = build_answerer({"kind": "openai_compat", "endpoint": args.endpoint,
                               "send_images": True, "max_images": args.k,
                               "model": "served", "prompt_style": "v3",
                               "timeout_seconds": 600})

    examples = [ex for ex in load_examples(args.qa) if ex.video_id in ledgers]
    out = Path(args.out)
    done = set()
    if out.exists():  # crash-safe resume
        for line in open(out):
            done.add(json.loads(line)["qid"])
    fh = open(out, "a")
    arms = ["greedy", "consensus", "tagged"]
    tally = {a: [] for a in arms}
    for i, ex in enumerate(examples):
        if ex.qid in done:
            continue
        L = ledgers[ex.video_id]
        # READ held fixed: one retrieval, shared by every arm
        frames = [f for f, _ in retriever.retrieve(
            ex.video_id, ex.question, args.k,
            query_embedding=query_embedding_for(ex))]
        rec = {"qid": ex.qid, "video_id": ex.video_id, "agreement": L["agreement"]}
        for a in arms:
            rec[a] = answer_with(answerer, ex, frames, L[a], tagged=(a == "tagged"))
            tally[a].append(rec[a])
        fh.write(json.dumps(rec) + "\n")
        fh.flush()
        if (i + 1) % 25 == 0:
            msg = " ".join(f"{a}={sum(tally[a])/max(1,len(tally[a])):.3f}" for a in arms)
            print(f"{i+1}/{len(examples)} | {msg}", flush=True)
    fh.close()
    if tally["greedy"]:
        msg = " ".join(f"{a}={sum(tally[a])/len(tally[a]):.3f}" for a in arms)
        print(f"FINAL n={len(tally['greedy'])} {msg}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
