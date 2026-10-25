"""JSON/JSONL IO helpers."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from .types import FrameRecord, QAExample


def ensure_dir(path: str | os.PathLike[str]) -> None:
    Path(path).mkdir(parents=True, exist_ok=True)


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    parent = Path(path).parent
    if str(parent):
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def read_jsonl(path: str | os.PathLike[str]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                rows.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSONL row") from exc
    return rows


def write_jsonl(path: str | os.PathLike[str], rows: Iterable[Dict[str, Any]]) -> None:
    parent = Path(path).parent
    if str(parent):
        ensure_dir(parent)
    with open(path, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")


def load_frames(path: str | os.PathLike[str]) -> List[FrameRecord]:
    return [FrameRecord.from_dict(row) for row in read_jsonl(path)]


def load_examples(path: str | os.PathLike[str]) -> List[QAExample]:
    return [QAExample.from_dict(row) for row in read_jsonl(path)]


def save_results(path: str | os.PathLike[str], results: Sequence[Any]) -> None:
    write_jsonl(path, [item.to_dict() if hasattr(item, "to_dict") else dict(item) for item in results])
