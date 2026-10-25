#!/usr/bin/env python3
"""Decode frames only for videos referenced by a JSONL QA split."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


VIDEO_EXTENSIONS = {".mp4", ".mkv", ".webm", ".avi", ".mov", ".m4v"}


def read_video_ids(qa_path: Path) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    with qa_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            video_id = str(row["video_id"])
            if video_id not in seen:
                ordered.append(video_id)
                seen.add(video_id)
    return ordered


def build_video_index(video_root: Path) -> dict[str, Path]:
    index: dict[str, Path] = {}
    for path in sorted(video_root.rglob("*")):
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            index.setdefault(path.stem, path)
            index.setdefault(path.name, path)
    return index


def resolve_video_id(video_id: str) -> str:
    """Normalize QA video_id to a directory-safe stem (strip extension)."""
    return Path(video_id).stem


def resolve_video_path(video_index: dict[str, Path], video_id: str) -> Path | None:
    return video_index.get(video_id) or video_index.get(resolve_video_id(video_id))


def count_frames(frame_dir: Path) -> int:
    if not frame_dir.exists():
        return 0
    return sum(1 for path in frame_dir.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"})


def decode_video(
    ffmpeg: str,
    video_path: Path,
    output_dir: Path,
    fps: float,
    jpeg_quality: int,
    overwrite: bool,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    existing = count_frames(output_dir)
    if existing and not overwrite:
        return {
            "video_path": str(video_path),
            "output_dir": str(output_dir),
            "returncode": 0,
            "status": "skipped_existing",
            "frame_count": existing,
            "seconds": 0.0,
        }
    if overwrite:
        for path in output_dir.glob("frame_*.jpg"):
            path.unlink()
    started = time.time()
    proc = subprocess.run(
        [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(video_path),
            "-vf",
            f"fps={fps}",
            "-q:v",
            str(jpeg_quality),
            str(output_dir / "frame_%06d.jpg"),
        ],
        text=True,
        capture_output=True,
    )
    return {
        "video_path": str(video_path),
        "output_dir": str(output_dir),
        "returncode": proc.returncode,
        "status": "decoded" if proc.returncode == 0 else "failed",
        "frame_count": count_frames(output_dir),
        "seconds": round(time.time() - started, 2),
        "stderr_tail": proc.stderr.splitlines()[-20:],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--qa-path", required=True)
    parser.add_argument("--video-root", required=True)
    parser.add_argument("--out-root", required=True)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--jpeg-quality", type=int, default=2)
    parser.add_argument("--ffmpeg", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--limit-videos", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--require-all", action="store_true")
    args = parser.parse_args()

    ffmpeg = args.ffmpeg or (".conda-env/bin/ffmpeg" if Path(".conda-env/bin/ffmpeg").exists() else "ffmpeg")
    if shutil.which(ffmpeg) is None and not Path(ffmpeg).exists():
        raise SystemExit(f"ffmpeg not found: {ffmpeg}")

    qa_path = Path(args.qa_path)
    video_root = Path(args.video_root)
    out_root = Path(args.out_root)
    out_root.mkdir(parents=True, exist_ok=True)

    video_ids = read_video_ids(qa_path)
    if args.limit_videos:
        video_ids = video_ids[: args.limit_videos]
    video_index = build_video_index(video_root)
    manifest: dict[str, Any] = {
        "qa_path": str(qa_path),
        "video_root": str(video_root),
        "out_root": str(out_root),
        "fps": args.fps,
        "requested_videos": len(video_ids),
        "decoded": [],
        "missing": [],
    }

    for idx, video_id in enumerate(video_ids, start=1):
        video_path = resolve_video_path(video_index, video_id)
        frame_key = resolve_video_id(video_id)
        if video_path is None:
            print(f"[{idx}/{len(video_ids)}] missing video {video_id}")
            manifest["missing"].append(video_id)
            continue
        print(f"[{idx}/{len(video_ids)}] decoding {video_id}")
        row = decode_video(
            ffmpeg=ffmpeg,
            video_path=video_path,
            output_dir=out_root / frame_key,
            fps=args.fps,
            jpeg_quality=args.jpeg_quality,
            overwrite=args.overwrite,
        )
        row["video_id"] = video_id
        manifest["decoded"].append(row)

    manifest["ok"] = (
        not manifest["missing"]
        and all(row["returncode"] == 0 and row["frame_count"] > 0 for row in manifest["decoded"])
    )
    manifest["total_frames"] = sum(int(row.get("frame_count", 0)) for row in manifest["decoded"])
    manifest_path = Path(args.manifest or f"runs/decode_{qa_path.stem}_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {manifest_path}; frames={manifest['total_frames']} missing={len(manifest['missing'])}")
    if args.require_all and not manifest["ok"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
