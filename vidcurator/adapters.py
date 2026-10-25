"""Benchmark normalizers.

These functions are conservative field mappers. They produce the project JSONL
schema and keep original rows under metadata for auditability.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io import write_jsonl


def load_raw_rows(path: str) -> List[Dict[str, Any]]:
    suffix = Path(path).suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        with open(path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            return payload
        for key in ("data", "examples", "annotations"):
            if isinstance(payload.get(key), list):
                return payload[key]
        raise ValueError(f"cannot find row list in {path}")
    if suffix == ".csv":
        with open(path, "r", encoding="utf-8", newline="") as handle:
            return list(csv.DictReader(handle))
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:  # pragma: no cover - environment diagnostic
            raise ValueError("parquet inputs require pandas/pyarrow") from exc
        return [_json_safe(row) for row in pd.read_parquet(path).to_dict("records")]
    raise ValueError(f"unsupported input format: {path}")


def normalize_dataset(dataset: str, input_path: str, output_path: str) -> None:
    rows = load_raw_rows(input_path)
    if dataset == "longvideobench":
        normalized = [_normalize_longvideobench(row, i) for i, row in enumerate(rows)]
    elif dataset == "videomme":
        normalized = [_normalize_videomme(row, i) for i, row in enumerate(rows)]
    elif dataset in {"nextgqa", "next-gqa"}:
        normalized = [_normalize_nextgqa(row, i) for i, row in enumerate(rows)]
    elif dataset in {"nextqa", "next-qa", "egoschema", "mlvu", "generic"}:
        normalized = [_normalize_generic(row, i, dataset) for i, row in enumerate(rows)]
    else:
        raise ValueError(f"unknown dataset adapter: {dataset}")
    write_jsonl(output_path, normalized)


def _normalize_longvideobench(row: Dict[str, Any], i: int) -> Dict[str, Any]:
    options = _options(row)
    return {
        "qid": str(row.get("id", row.get("qid", f"lvb-{i}"))),
        "video_id": str(row.get("video_id", row.get("video", row.get("video_path", "")))),
        "question": str(row.get("question", row.get("query", ""))),
        "options": options,
        "answer": str(row.get("answer", "")),
        "answer_index": _answer_index(row, options),
        "evidence_spans": _spans(row),
        "domain": "longvideobench",
        "split": str(row.get("split", "eval")),
        "metadata": {"source": "longvideobench", "raw": row},
    }


def _normalize_videomme(row: Dict[str, Any], i: int) -> Dict[str, Any]:
    options = _options(row)
    return {
        "qid": str(row.get("question_id", row.get("id", f"videomme-{i}"))),
        "video_id": str(row.get("video_id", row.get("url", row.get("video", "")))),
        "question": str(row.get("question", "")),
        "options": options,
        "answer": str(row.get("answer", "")),
        "answer_index": _answer_index(row, options),
        "evidence_spans": _spans(row),
        "domain": str(row.get("domain", "videomme")),
        "split": str(row.get("split", "eval")),
        "metadata": {"source": "videomme", "raw": row},
    }


def _normalize_nextgqa(row: Dict[str, Any], i: int) -> Dict[str, Any]:
    options = _options(row)
    return {
        "qid": str(row.get("qid", row.get("question_id", f"nextgqa-{i}"))),
        "video_id": str(row.get("video_id", row.get("video", ""))),
        "question": str(row.get("question", "")),
        "options": options,
        "answer": str(row.get("answer", "")),
        "answer_index": _answer_index(row, options),
        "evidence_spans": _spans(row),
        "domain": "nextgqa",
        "split": str(row.get("split", "eval")),
        "metadata": {"source": "nextgqa", "raw": row},
    }


def _normalize_generic(row: Dict[str, Any], i: int, dataset: str) -> Dict[str, Any]:
    options = _options(row)
    return {
        "qid": str(row.get("qid", row.get("question_id", row.get("id", f"{dataset}-{i}")))),
        "video_id": str(row.get("video_id", row.get("video", row.get("video_path", "")))),
        "question": str(row.get("question", row.get("query", ""))),
        "options": options,
        "answer": str(row.get("answer", "")),
        "answer_index": _answer_index(row, options),
        "evidence_spans": _spans(row),
        "domain": str(row.get("domain", dataset)),
        "split": str(row.get("split", "eval")),
        "metadata": {"source": dataset, "raw": row},
    }


def _options(row: Dict[str, Any]) -> List[str]:
    value = row.get("options", row.get("candidates", row.get("choices", [])))
    if isinstance(value, dict):
        return [str(value[key]) for key in sorted(value)]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return [str(x) for x in parsed]
        except json.JSONDecodeError:
            return [part.strip() for part in value.split("|") if part.strip()]
    if value is not None:
        try:
            values = value.tolist()
        except AttributeError:
            values = value
        if isinstance(values, (list, tuple)):
            return [str(x) for x in values]
        if values:
            return [str(values)]
    if value:
        return [str(x) for x in value]
    numbered = []
    for idx in range(10):
        key = f"a{idx}"
        if key in row and row[key] not in (None, ""):
            numbered.append(str(row[key]))
    return numbered


def _answer_index(row: Dict[str, Any], options: List[str]) -> int | None:
    for key in ("answer_index", "answer_idx", "correct_idx", "correct_choice", "label"):
        if key in row and row[key] not in (None, ""):
            value = row[key]
            if isinstance(value, str) and value.upper() in {"A", "B", "C", "D", "E"}:
                return ord(value.upper()) - ord("A")
            return int(value)
    answer = str(row.get("answer", ""))
    if answer.upper() in {"A", "B", "C", "D", "E"}:
        return ord(answer.upper()) - ord("A")
    for idx, option in enumerate(options):
        if answer and answer.strip().lower() == option.strip().lower():
            return idx
    return None


def _json_safe(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_json_safe(v) for v in value]
    return value


def _spans(row: Dict[str, Any]) -> List[Dict[str, Any]]:
    candidates: Iterable[Any] = (
        row.get("evidence_spans")
        or row.get("temporal_spans")
        or row.get("relevant_windows")
        or row.get("moment")
        or []
    )
    if isinstance(candidates, str):
        try:
            candidates = json.loads(candidates)
        except json.JSONDecodeError:
            return []
    if isinstance(candidates, dict):
        candidates = [candidates]
    spans = []
    for item in candidates:
        if isinstance(item, (list, tuple)) and len(item) >= 2:
            spans.append({"start": float(item[0]), "end": float(item[1]), "label": ""})
        elif isinstance(item, dict):
            start = item.get("start", item.get("start_time", item.get("s", 0.0)))
            end = item.get("end", item.get("end_time", item.get("e", start)))
            spans.append({"start": float(start), "end": float(end), "label": str(item.get("label", ""))})
    return spans
