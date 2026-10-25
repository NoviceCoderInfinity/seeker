#!/usr/bin/env python3
"""Download the approved priority assets for VID-CURATOR.

Authentication is read from the normal Hugging Face mechanisms, especially the
HF_TOKEN environment variable or a prior `huggingface-cli login`.
"""

from __future__ import annotations

import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Iterable


MODEL_PRIORITY = [
    "qwen2_5_vl_7b",
    "qwen3_vl_8b_instruct",
    "internvl3_5_8b",
    "llava_video_7b_qwen2",
]

DATASET_PRIORITY = [
    "nextgqa",
    "videomme_long",
    "longvideobench",
    "mlvu",
    "nextqa_temporal",
    "egoschema",
]


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def by_name(rows: Iterable[dict]) -> dict[str, dict]:
    return {row["name"]: row for row in rows}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", action="store_true")
    parser.add_argument("--datasets", action="store_true")
    parser.add_argument("--metadata-only", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--allow-large-video", action="store_true")
    parser.add_argument("--only", nargs="*", default=[], help="Limit to specific manifest names.")
    parser.add_argument("--manifest", default="runs/download_priority_assets_manifest.json")
    args = parser.parse_args()

    if not args.models and not args.datasets:
        args.models = True
        args.datasets = True

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first") from exc

    token = os.environ.get("HF_TOKEN")
    models = by_name(load_json("manifests/models.json")["models"])
    datasets = by_name(load_json("manifests/datasets.json")["datasets"])

    jobs = []
    if args.models:
        for name in MODEL_PRIORITY:
            row = models[name]
            local_dir = Path("models") / name
            allow = None
            ignore = None
            if args.metadata_only:
                allow = ["*.json", "*.txt", "*.md", "*.py", "*.model", "*.tiktoken"]
            jobs.append((name, row["hf_id"], "model", local_dir, allow, ignore))
    if args.datasets:
        for name in DATASET_PRIORITY:
            row = datasets[name]
            local_dir = Path(row["raw_dir"])
            allow = None
            ignore = None
            if args.metadata_only or not args.allow_large_video:
                allow = [
                    "*.json",
                    "*.jsonl",
                    "*.csv",
                    "*.parquet",
                    "*.txt",
                    "*.md",
                    "README*",
                    "LICENSE*",
                    "subtitle.zip",
                    "datasets/**",
                    "annotations/**",
                ]
                ignore = ["*.mp4", "*.mkv", "*.webm", "*.mov", "*.zip.*", "videos_chunked_*.zip"]
            jobs.append((name, row["hf_id"], row.get("hf_type", "dataset"), local_dir, allow, ignore))
    if args.only:
        keep = set(args.only)
        jobs = [job for job in jobs if job[0] in keep]
        missing = sorted(keep - {job[0] for job in jobs})
        if missing:
            raise SystemExit(f"unknown --only asset(s): {', '.join(missing)}")

    manifest: dict[str, list[dict]] = {"completed": [], "failed": [], "skipped": []}
    for name, hf_id, repo_type, local_dir, allow, ignore in jobs:
        if args.skip_existing and local_dir.exists() and any(local_dir.iterdir()):
            print(f"skip existing {name}: {local_dir}")
            manifest["skipped"].append({"name": name, "local_dir": str(local_dir), "reason": "exists"})
            continue
        print(f"download {name}: {hf_id} -> {local_dir} ({repo_type})")
        if args.dry_run:
            manifest["skipped"].append({"name": name, "local_dir": str(local_dir), "reason": "dry_run"})
            continue
        local_dir.mkdir(parents=True, exist_ok=True)
        try:
            snapshot_download(
                repo_id=hf_id,
                repo_type=repo_type,
                local_dir=str(local_dir),
                token=token,
                allow_patterns=allow,
                ignore_patterns=ignore,
            )
            manifest["completed"].append({"name": name, "hf_id": hf_id, "local_dir": str(local_dir)})
        except Exception as exc:  # pragma: no cover - operational script
            manifest["failed"].append(
                {
                    "name": name,
                    "hf_id": hf_id,
                    "local_dir": str(local_dir),
                    "error": str(exc),
                    "traceback_tail": traceback.format_exc().splitlines()[-8:],
                }
            )
            print(f"failed {name}: {exc}")
    out = Path(args.manifest)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"wrote {out}")
    if manifest["failed"]:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
