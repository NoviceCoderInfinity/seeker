#!/usr/bin/env python3
"""Build a VID-CURATOR frame JSONL index from decoded frame directories.

The script is intentionally usable before embeddings/captions exist. It writes
timestamps, paths, and metadata now; a later embedding/caption pass can enrich
the same schema without changing experiment configs.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


FRAME_RE = re.compile(r"(\d+)")


def infer_frame_number(path: Path) -> int:
    matches = FRAME_RE.findall(path.stem)
    if not matches:
        return 0
    return int(matches[-1])


def iter_frames(root: Path, extensions: set[str]):
    for video_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        video_id = video_dir.name
        for frame_path in sorted(video_dir.iterdir()):
            if frame_path.suffix.lower() in extensions:
                yield video_id, frame_path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--frames-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--fps", type=float, required=True)
    parser.add_argument("--dataset", default="")
    parser.add_argument("--path-prefix", default="")
    parser.add_argument("--caption-template", default="{dataset} frame at {timestamp:.1f}s")
    args = parser.parse_args()

    frames_root = Path(args.frames_root)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    extensions = {".jpg", ".jpeg", ".png", ".webp"}
    rows = 0
    with out.open("w", encoding="utf-8") as handle:
        for video_id, frame_path in iter_frames(frames_root, extensions):
            frame_num = infer_frame_number(frame_path)
            timestamp = max(0.0, (frame_num - 1) / args.fps)
            path = str(frame_path)
            if args.path_prefix:
                path = str(Path(args.path_prefix) / frame_path.relative_to(frames_root))
            row = {
                "video_id": video_id,
                "frame_id": f"{frame_num:06d}",
                "timestamp": timestamp,
                "path": path,
                "caption": args.caption_template.format(
                    dataset=args.dataset or frames_root.name,
                    video_id=video_id,
                    frame_id=f"{frame_num:06d}",
                    timestamp=timestamp,
                ),
                "embedding": None,
                "metadata": {
                    "dataset": args.dataset,
                    "fps": args.fps,
                    "source": "decoded_frames",
                },
            }
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
            rows += 1
    print(f"wrote {rows} frame rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
