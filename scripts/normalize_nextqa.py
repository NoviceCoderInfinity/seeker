#!/usr/bin/env python3
"""Normalize NExT-QA multiple-choice rows into VID-CURATOR JSONL splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


TEMPORAL_CAUSAL_TYPES = {"TN", "TC", "TP", "CW", "CH"}


def normalize_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    options = [str(row[f"a{i}"]) for i in range(5) if row.get(f"a{i}") is not None]
    answer_index = int(row["answer"]) if row.get("answer") is not None else None
    answer = options[answer_index] if answer_index is not None and 0 <= answer_index < len(options) else ""
    video_id = str(row["video"])
    qid_raw = str(row["qid"])
    qtype = str(row.get("type", ""))
    return {
        "qid": f"{video_id}_{qid_raw}",
        "video_id": video_id,
        "question": str(row.get("question", "")),
        "options": options,
        "answer": answer,
        "answer_index": answer_index,
        "evidence_spans": [],
        "domain": f"nextqa_{qtype.lower()}",
        "split": split,
        "metadata": {
            "source": "nextqa",
            "source_qid": qid_raw,
            "type": qtype,
            "frame_count": int(row["frame_count"]) if row.get("frame_count") is not None else None,
            "width": int(row["width"]) if row.get("width") is not None else None,
            "height": int(row["height"]) if row.get("height") is not None else None,
            "video_path_hint": f"NExTVideo/{video_id}.mp4",
        },
    }


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/raw/nextqa/MC/test-00000-of-00001.parquet")
    # NExT-QA MC ships only the test split on HF (lmms-lab/NExTQA); we report on
    # MC test temporal/causal (7159 Q). Name files 'test' so the split is unambiguous.
    parser.add_argument("--output", default="data/processed/nextqa_temporal_test.jsonl")
    parser.add_argument("--small-output", default="data/processed/nextqa_temporal_test_150.jsonl")
    parser.add_argument("--small-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split", default="test")
    parser.add_argument("--include-types", nargs="*", default=sorted(TEMPORAL_CAUSAL_TYPES))
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("normalize_nextqa.py requires pandas and pyarrow") from exc

    include_types = set(args.include_types)
    df = pd.read_parquet(args.input)
    rows = [
        normalize_row(row, args.split)
        for row in df.to_dict("records")
        if str(row.get("type", "")) in include_types
    ]
    small = list(rows)
    random.Random(args.seed).shuffle(small)
    small = small[: args.small_size]
    write_jsonl(Path(args.output), rows)
    write_jsonl(Path(args.small_output), small)
    print(f"wrote {len(rows)} temporal/causal rows to {args.output}")
    print(f"wrote {len(small)} small rows to {args.small_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
