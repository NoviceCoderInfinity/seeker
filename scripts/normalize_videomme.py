#!/usr/bin/env python3
"""Normalize Video-MME parquet metadata into VID-CURATOR JSONL splits."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [json_safe(v) for v in value]
    return value


def answer_index(answer: Any) -> int | None:
    if answer is None:
        return None
    text = str(answer).strip()
    if not text:
        return None
    if text.upper() in {"A", "B", "C", "D", "E"}:
        return ord(text.upper()) - ord("A")
    if text.isdigit():
        return int(text)
    return None


def normalize_row(row: dict[str, Any], split: str) -> dict[str, Any]:
    raw = json_safe(row)
    options = [str(item) for item in json_safe(row.get("options", []))]
    idx = answer_index(row.get("answer"))
    video_youtube_id = str(row.get("videoID") or row.get("video_id"))
    answer_text = options[idx] if idx is not None and 0 <= idx < len(options) else str(row.get("answer", ""))
    return {
        "qid": str(row.get("question_id")),
        "video_id": video_youtube_id,
        "question": str(row.get("question", "")),
        "options": options,
        "answer": answer_text,
        "answer_index": idx,
        "evidence_spans": [],
        "domain": str(row.get("domain", "videomme")),
        "split": split,
        "metadata": {
            "source": "videomme",
            "duration": str(row.get("duration", "")),
            "sub_category": str(row.get("sub_category", "")),
            "task_type": str(row.get("task_type", "")),
            "numeric_video_id": str(row.get("video_id", "")),
            "url": str(row.get("url", "")),
            "video_path_hint": f"data/{video_youtube_id}.mp4",
            "raw": raw,
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
    parser.add_argument("--input", default="data/raw/videomme/videomme/test-00000-of-00001.parquet")
    parser.add_argument("--output-all", default="data/processed/videomme_all.jsonl")
    parser.add_argument("--output-long", default="data/processed/videomme_long.jsonl")
    parser.add_argument("--small-output", default="data/processed/videomme_long_150.jsonl")
    parser.add_argument("--small-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--split", default="test")
    args = parser.parse_args()

    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("normalize_videomme.py requires pandas and pyarrow") from exc

    df = pd.read_parquet(args.input)
    all_rows = [normalize_row(row, args.split) for row in df.to_dict("records")]
    long_rows = [row for row in all_rows if row["metadata"].get("duration") == "long"]
    small_rows = list(long_rows)
    random.Random(args.seed).shuffle(small_rows)
    small_rows = small_rows[: args.small_size]

    write_jsonl(Path(args.output_all), all_rows)
    write_jsonl(Path(args.output_long), long_rows)
    write_jsonl(Path(args.small_output), small_rows)
    print(f"wrote {len(all_rows)} rows to {args.output_all}")
    print(f"wrote {len(long_rows)} long rows to {args.output_long}")
    print(f"wrote {len(small_rows)} small rows to {args.small_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
