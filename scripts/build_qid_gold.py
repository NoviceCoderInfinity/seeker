#!/usr/bin/env python3
"""Build a qid -> gold answer_index map for the outcome-aware GRPO reward.

The SFT/RL traces carry only {qid, hop, action, policy} in metadata (no gold),
so the outcome reward looks gold up by qid from the processed QA splits.
"""

from __future__ import annotations

import json
import glob
from pathlib import Path


def main() -> int:
    gold: dict[str, int] = {}
    for p in sorted(glob.glob("data/processed/*.jsonl")):
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                qid = r.get("qid") or r.get("id") or r.get("question_id")
                gi = r.get("answer_index")
                if gi is None:
                    gi = r.get("gold_index")
                if qid is not None and gi is not None:
                    gold[str(qid)] = int(gi)
    out = Path("data/rl/qid_gold.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(gold), encoding="utf-8")
    print(f"wrote {out} ({len(gold)} qids)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
