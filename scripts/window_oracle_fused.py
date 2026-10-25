#!/usr/bin/env python3
"""Gate 0 (plan9): FUSED multi-stream temporal-window oracle on long video.

Extends scripts/window_oracle.py from frames-only to a fused subtitle+frame store.
For each contiguous temporal window we answer THREE ways:
  - frame-only : uniform frames of the window          (reproduces the 0.673 any-window ceiling)
  - subs-only  : that window's subtitle span, 0 frames (locality of the transcript stream)
  - fused      : window frames  +  window subtitle span (the plan9 substrate)
plus one video-level subtitle-only answer (all subs, 0 frames) to anchor the 0.548 subs-only prior.

Decision (Gate 0):
  * KILL  if fused-any-window oracle <= max(frame-any, subs-only-video) + noise
           -> no cross-modal headroom, ship training-free multi-stream system, do NOT train a controller.
  * GO    if fused-any-window > frame-any AND subs/frame contributions are complementary
           (windows where fused is right but neither single stream is) -> real learnable READ target.

Usage (inside a job serving Qwen on $PORT):
  python scripts/window_oracle_fused.py \
    --qa data/processed/videomme_long_450.jsonl \
    --frames data/indices/videomme_long_decoded_frames.jsonl \
    --subtitle-dir data/subtitles/videomme \
    --endpoint http://127.0.0.1:8000 --windows 8 --per-window 8 \
    --subtitle-chars 3500 --limit 24 \
    --out reports/window_oracle_fused_videomme_pilot.md
"""
from __future__ import annotations
import argparse, json
from pathlib import Path

from vidcurator.io import load_frames, load_examples
from vidcurator.retrieval import FrameIndex
from vidcurator.subtitles import load_subtitles
from vidcurator.answerers import build_answerer


def _render_subs(entries, char_budget):
    """Render [(t,text)] as timestamped ledger lines, evenly downsampled to a char budget."""
    if not entries:
        return ""
    lines = [f"[t={int(t)}s sub] {txt}" for t, txt in entries]
    blob = "\n".join(lines)
    if len(blob) <= char_budget or len(lines) <= 1:
        return blob[:char_budget]
    # even temporal downsample (never head-cut)
    keep = max(1, int(len(lines) * char_budget / len(blob)))
    step = max(1, len(lines) // keep)
    return "\n".join(lines[::step])[:char_budget]


def _ctx(ex, sub_text=None):
    opts = ex.options or []
    ctx = f"Question: {ex.question}"
    if opts:
        ctx += "\nOptions:\n" + "\n".join(f"{chr(ord('A') + i)}. {o}" for i, o in enumerate(opts))
    if sub_text:
        ctx = "Transcript excerpt:\n" + sub_text + "\n\n" + ctx
    return ctx


def _pick(chunk, per_window):
    step = max(1, len(chunk) // per_window)
    return chunk[::step][:per_window]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--qa", required=True)
    ap.add_argument("--frames", required=True)
    ap.add_argument("--subtitle-dir", required=True)
    ap.add_argument("--endpoint", required=True)
    ap.add_argument("--windows", type=int, default=8)
    ap.add_argument("--per-window", type=int, default=8)
    ap.add_argument("--subtitle-chars", type=int, default=3500)
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
        frames = sorted(frames, key=lambda f: f.timestamp)
        n = len(frames)
        subs = load_subtitles(ex.video_id, args.subtitle_dir)
        row = {"qid": ex.qid, "gold": ex.answer_index,
               "frame_only": [], "subs_only": [], "fused": [], "subs_video": None}

        # video-level subtitle-only anchor (0.548 prior)
        try:
            p = answerer.answer(ex, [], _ctx(ex, _render_subs(subs, args.subtitle_chars * 3)))
            row["subs_video"] = p.answer_index
        except Exception as e:
            print(f"  ! qid={ex.qid} subs_video: {e}", flush=True)

        for w in range(W):
            chunk = frames[w * n // W:(w + 1) * n // W]
            if not chunk:
                row["frame_only"].append(None); row["subs_only"].append(None); row["fused"].append(None)
                continue
            t0, t1 = chunk[0].timestamp, chunk[-1].timestamp
            win_subs = [(t, x) for (t, x) in subs if t0 <= t <= t1] if subs else []
            sub_text = _render_subs(win_subs, args.subtitle_chars)
            picked = _pick(chunk, args.per_window)
            for name, imgs, stext in (("frame_only", picked, None),
                                      ("subs_only", [], sub_text or None),
                                      ("fused", picked, sub_text or None)):
                try:
                    pred = answerer.answer(ex, imgs, _ctx(ex, stext))
                    row[name].append(pred.answer_index)
                except Exception as e:
                    print(f"  ! qid={ex.qid} w={w} {name}: {e}", flush=True)
                    row[name].append(None)
        per_q.append(row)
        if len(per_q) % 10 == 0:
            print(f"{len(per_q)} questions done", flush=True)
            _flush(per_q, W, args)  # crash-safe checkpoint (survives slurm wall timeout)

    _flush(per_q, W, args)
    return 0


def _flush(per_q, W, args):
    n = len(per_q)

    def any_oracle(key):
        return sum(1 for r in per_q if any(
            a is not None and int(a) == r["gold"] for a in r[key])) / max(n, 1)

    def best_fixed(key):
        best = 0.0
        for w in range(W):
            ok = sum(1 for r in per_q if r[key][w] is not None and int(r[key][w]) == r["gold"])
            best = max(best, ok / max(n, 1))
        return best

    subs_video_acc = sum(1 for r in per_q if r["subs_video"] is not None
                         and int(r["subs_video"]) == r["gold"]) / max(n, 1)

    # complementarity: fused-window right where BOTH frame-window and subs-window (same w) are wrong
    comp = 0
    for r in per_q:
        for w in range(W):
            f, s, fu = r["frame_only"][w], r["subs_only"][w], r["fused"][w]
            if fu is not None and int(fu) == r["gold"]:
                fw = f is not None and int(f) == r["gold"]
                sw = s is not None and int(s) == r["gold"]
                if not fw and not sw:
                    comp += 1
                    break

    frame_any, subs_any, fused_any = any_oracle("frame_only"), any_oracle("subs_only"), any_oracle("fused")
    single_ceiling = max(frame_any, subs_video_acc)
    lines = [
        f"# FUSED window oracle (Gate 0) — {Path(args.qa).stem} (W={W}, {args.per_window} f/win, n={n})",
        "",
        "| stream | best fixed window | ANY-window oracle |",
        "|---|---|---|",
        f"| frame-only | {best_fixed('frame_only'):.3f} | {frame_any:.3f} |",
        f"| subs-only (window) | {best_fixed('subs_only'):.3f} | {subs_any:.3f} |",
        f"| **fused** | {best_fixed('fused'):.3f} | **{fused_any:.3f}** |",
        "",
        f"- subtitle-only (whole video, 0 frames) anchor: {subs_video_acc:.3f}",
        f"- single-stream ceiling max(frame-any, subs-video): {single_ceiling:.3f}",
        f"- **fused lift over single-stream ceiling: {fused_any - single_ceiling:+.3f}**",
        f"- complementary questions (fused-window right, both single streams wrong): "
        f"{comp}/{n} = {comp / max(n, 1):.3f}",
        "",
        "## Gate 0 verdict",
        f"- {'GO' if fused_any > single_ceiling + 0.03 and comp / max(n, 1) >= 0.05 else 'LEAN-KILL'}: "
        f"fused_any={fused_any:.3f} vs single_ceiling={single_ceiling:.3f}, complementarity={comp / max(n, 1):.3f}",
        "  (pilot n is small — treat as smoke signal, not the decision; scale to n=450 if GO-leaning.)",
    ]
    Path(args.out).write_text("\n".join(lines) + "\n")
    Path(args.out).with_suffix(".per_question.jsonl").write_text(
        "\n".join(json.dumps(r) for r in per_q) + "\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
