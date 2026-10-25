#!/usr/bin/env python3
"""Idempotently extract downloaded benchmark video zip chunks."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import time
from pathlib import Path
from typing import Any


def count_files(root: Path, suffixes: set[str]) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def extract_archive(archive: Path, output_dir: Path) -> dict[str, Any]:
    before = count_files(output_dir, {".mp4", ".mkv", ".webm", ".avi", ".mov"})
    started = time.time()
    proc = subprocess.run(
        ["unzip", "-n", str(archive), "-d", str(output_dir)],
        text=True,
        capture_output=True,
    )
    after = count_files(output_dir, {".mp4", ".mkv", ".webm", ".avi", ".mov"})
    return {
        "archive": str(archive),
        "output_dir": str(output_dir),
        "returncode": proc.returncode,
        "seconds": round(time.time() - started, 2),
        "video_count_before": before,
        "video_count_after": after,
        "new_video_count_estimate": max(0, after - before),
        "stdout_tail": proc.stdout.splitlines()[-20:],
        "stderr_tail": proc.stderr.splitlines()[-20:],
    }


def default_archives(dataset: str) -> list[Path]:
    if dataset == "videomme":
        return sorted(Path("data/raw/videomme").glob("videos_chunked_*.zip"))
    if dataset == "egoschema":
        return sorted(Path("data/raw/egoschema").glob("videos_chunked_*.zip"))
    if dataset == "nextqa":
        return [Path("data/raw/nextqa/videos.zip")]
    raise ValueError(f"unknown default dataset: {dataset}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["videomme", "egoschema", "nextqa"], required=True)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--manifest", default="")
    parser.add_argument("--archives", nargs="*", default=None)
    parser.add_argument("--fail-fast", action="store_true")
    args = parser.parse_args()

    if shutil.which("unzip") is None:
        raise SystemExit("unzip is required for archive extraction")

    output_dir = Path(args.output_dir or f"data/videos/{args.dataset}")
    output_dir.mkdir(parents=True, exist_ok=True)
    archives = [Path(path) for path in args.archives] if args.archives else default_archives(args.dataset)
    if not archives:
        raise SystemExit(f"no archives found for {args.dataset}")

    manifest = {
        "dataset": args.dataset,
        "output_dir": str(output_dir),
        "archives": [],
        "video_count_before": count_files(output_dir, {".mp4", ".mkv", ".webm", ".avi", ".mov"}),
    }
    for archive in archives:
        print(f"extracting {archive} -> {output_dir}")
        row = extract_archive(archive, output_dir)
        manifest["archives"].append(row)
        if row["returncode"] != 0 and args.fail_fast:
            break
    manifest["video_count_after"] = count_files(output_dir, {".mp4", ".mkv", ".webm", ".avi", ".mov"})
    manifest["ok"] = all(row["returncode"] == 0 for row in manifest["archives"])

    manifest_path = Path(args.manifest or f"runs/extract_{args.dataset}_manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {manifest_path}; videos now {manifest['video_count_after']}")
    return 0 if manifest["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
