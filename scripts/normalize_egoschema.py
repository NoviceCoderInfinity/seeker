#!/usr/bin/env python3
"""Normalize EgoSchema parquet metadata into VID-CURATOR JSONL splits."""

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
    if not text or text.lower() == "nan":
        return None
    if text.upper() in {"A", "B", "C", "D", "E"}:
        return ord(text.upper()) - ord("A")
    if text.isdigit():
        return int(text)
    return None


def normalize_row(row: dict[str, Any], split: str, source_split: str) -> dict[str, Any]:
    raw = json_safe(row)
    options = [str(item) for item in json_safe(row.get("option", row.get("options", [])))]
    idx = answer_index(row.get("answer"))
    answer_text = options[idx] if idx is not None and 0 <= idx < len(options) else ""
    return {
        "qid": str(row.get("question_idx")),
        "video_id": str(row.get("video_idx")),
        "question": str(row.get("question", "")),
        "options": options,
        "answer": answer_text,
        "answer_index": idx,
        "evidence_spans": [],
        "domain": "egoschema",
        "split": split,
        "metadata": {
            "source": "egoschema",
            "source_split": source_split,
            "video_path_hint": f"videos/{row.get('video_idx')}.mp4",
            "raw": raw,
        },
    }


def read_parquet(path: str) -> list[dict[str, Any]]:
    try:
        import pandas as pd
    except ImportError as exc:
        raise SystemExit("normalize_egoschema.py requires pandas and pyarrow") from exc
    return pd.read_parquet(path).to_dict("records")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--subset-input", default="data/raw/egoschema/Subset/test-00000-of-00001.parquet")
    parser.add_argument("--hidden-input", default="data/raw/egoschema/MC/test-00000-of-00001.parquet")
    parser.add_argument("--output", default="data/processed/egoschema_val.jsonl")
    parser.add_argument("--small-output", default="data/processed/egoschema_val_150.jsonl")
    parser.add_argument("--hidden-output", default="data/processed/egoschema_test_hidden.jsonl")
    parser.add_argument("--small-size", type=int, default=150)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    labeled = [normalize_row(row, "val", "Subset") for row in read_parquet(args.subset_input)]
    hidden = [normalize_row(row, "test_hidden", "MC") for row in read_parquet(args.hidden_input)]
    small = list(labeled)
    random.Random(args.seed).shuffle(small)
    small = small[: args.small_size]

    write_jsonl(Path(args.output), labeled)
    write_jsonl(Path(args.small_output), small)
    write_jsonl(Path(args.hidden_output), hidden)
    print(f"wrote {len(labeled)} labeled rows to {args.output}")
    print(f"wrote {len(small)} small rows to {args.small_output}")
    print(f"wrote {len(hidden)} hidden-test rows to {args.hidden_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
