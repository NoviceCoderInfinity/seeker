#!/usr/bin/env python3
"""Normalize NExT-GQA CSV plus temporal grounding files."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def answer_index(answer: str, options: list[str]) -> int | None:
    for idx, option in enumerate(options):
        if answer.strip().lower() == option.strip().lower():
            return idx
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--grounding-json", required=True)
    parser.add_argument("--frame2time-json", required=True)
    parser.add_argument("--output-qa", required=True)
    parser.add_argument("--output-frames", required=True)
    parser.add_argument("--split", default="val")
    args = parser.parse_args()

    grounding = load_json(args.grounding_json)
    frame2time = load_json(args.frame2time_json)

    qa_rows = []
    with open(args.csv, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            options = [row[f"a{i}"] for i in range(5) if row.get(f"a{i}", "") != ""]
            qid_raw = str(row["qid"])
            video_id = str(row["video_id"])
            spans = []
            for start, end in grounding.get(video_id, {}).get("location", {}).get(qid_raw, []):
                spans.append({"start": float(start), "end": float(end), "label": f"nextgqa qid {qid_raw}"})
            qa_rows.append(
                {
                    "qid": f"{video_id}_{qid_raw}",
                    "video_id": video_id,
                    "question": row["question"],
                    "options": options,
                    "answer": row["answer"],
                    "answer_index": answer_index(row["answer"], options),
                    "evidence_spans": spans,
                    "domain": "nextgqa",
                    "split": args.split,
                    "metadata": {
                        "source_qid": qid_raw,
                        "type": row.get("type", ""),
                        "frame_count": int(row["frame_count"]) if row.get("frame_count") else None,
                        "width": int(row["width"]) if row.get("width") else None,
                        "height": int(row["height"]) if row.get("height") else None,
                    },
                }
            )

    frame_rows = []
    for video_id, timestamps in frame2time.items():
        for idx, timestamp in enumerate(timestamps):
            frame_rows.append(
                {
                    "video_id": video_id,
                    "frame_id": f"{idx:06d}",
                    "timestamp": float(timestamp),
                    "path": f"data/frames/nextgqa/{video_id}/frame_{idx:06d}.jpg",
                    "caption": f"NExT-GQA timestamp frame at {float(timestamp):.1f}s",
                    "embedding": None,
                    "metadata": {"source": "frame2time"},
                }
            )

    Path(args.output_qa).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_qa, "w", encoding="utf-8") as handle:
        for row in qa_rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    Path(args.output_frames).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_frames, "w", encoding="utf-8") as handle:
        for row in frame_rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")

    print(f"wrote {len(qa_rows)} QA rows to {args.output_qa}")
    print(f"wrote {len(frame_rows)} frame rows to {args.output_frames}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
