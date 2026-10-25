#!/usr/bin/env python3
"""P1.1 — Freeze the train/test firewall for the learned curator.

The GRPO/RFT curator is trained on scripted-curator traces exported from the val files
(`data/rl/vidcurator_train_sft.jsonl`). Those trace qids (the TRAIN pool) must be walled off
from evaluation. We define, per evaluated subset, a STRICT held-out eval set = subset questions
whose qid is NOT a train qid AND whose video is NOT a train video (no question- and no
video-level leakage). Writes data/rl/firewall_split.json as the single source of truth read by
significance_tests.py and the clean eval jobs.

Run: python3 scripts/build_firewall_split.py
"""
from __future__ import annotations
import json, glob
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# evaluated subsets that already have uniform/scripted baselines (reused for clean eval)
EVAL_SUBSETS = {
    "nextgqa":       "data/processed/nextgqa_val_450.jsonl",
    "nextqa":        "data/processed/nextqa_temporal_val_450.jsonl",
    "videomme":      "data/processed/videomme_long_450.jsonl",
    "longvideobench":"data/processed/longvideobench_val_half.jsonl",
    "egoschema":     "data/processed/egoschema_val_half.jsonl",
    "mlvu":          "data/processed/mlvu_val_half.jsonl",
    "lvbench":       "data/processed/lvbench_val_siglip.jsonl",
}

def main() -> int:
    # Exclude BOTH the GRPO train split AND its internal-val split (used for checkpoint
    # selection) — plus every video either touches — so held-out eval is leakage-free.
    train_qids = set()
    for src in ("data/rl/vidcurator_train_sft.jsonl", "data/rl/vidcurator_val_sft.jsonl"):
        p = ROOT / src
        if not p.exists():
            continue
        for l in open(p):
            try:
                train_qids.add(str(json.loads(l).get("metadata", {}).get("qid")))
            except Exception:
                pass
    # qid -> video across every processed file (to recover train videos + subset videos)
    qid2vid = {}
    for f in glob.glob(str(ROOT / "data/processed/*.jsonl")):
        for l in open(f):
            try:
                r = json.loads(l); qid2vid[str(r["qid"])] = str(r.get("video_id"))
            except Exception:
                pass
    train_vids = {qid2vid[q] for q in train_qids if qid2vid.get(q)}

    split = {
        "_meta": {
            "policy": "STRICT held-out: eval qid not in (GRPO train+val) qids AND video not in any of their videos",
            "n_train_qids": len(train_qids), "n_train_videos": len(train_vids),
        },
        "train_qids": sorted(train_qids),
        "train_videos": sorted(train_vids),
        "eval": {},
    }
    print(f"train qids={len(train_qids)} train videos={len(train_vids)}\n")
    print(f"{'dataset':14s} {'subset':>6s} {'held-out(strict)':>16s}")
    for ds, f in EVAL_SUBSETS.items():
        rows = [json.loads(l) for l in open(ROOT / f)]
        ho = [str(r["qid"]) for r in rows
              if str(r["qid"]) not in train_qids and str(r.get("video_id")) not in train_vids]
        split["eval"][ds] = {"file": f, "subset_n": len(rows), "heldout_qids": ho, "heldout_n": len(ho)}
        print(f"{ds:14s} {len(rows):6d} {len(ho):16d}")

    out = ROOT / "data/rl/firewall_split.json"
    out.write_text(json.dumps(split, indent=1))
    print(f"\nWROTE {out}")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
