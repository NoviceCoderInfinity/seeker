#!/usr/bin/env python3
"""Download selected Hugging Face models or datasets from manifests.

This script is intentionally explicit. Use --dry-run first, then download one
asset at a time after license/auth checks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def load_json(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def find_asset(kind: str, name: str) -> dict:
    if kind == "model":
        rows = load_json("manifests/models.json")["models"]
    elif kind == "dataset":
        rows = load_json("manifests/datasets.json")["datasets"]
    else:
        raise ValueError(f"unknown kind: {kind}")
    for row in rows:
        if row["name"] == name:
            return row
    raise KeyError(f"{kind} not found in manifest: {name}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["model", "dataset"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--revision", default=None)
    parser.add_argument("--allow-pattern", action="append", default=[])
    parser.add_argument("--ignore-pattern", action="append", default=[])
    args = parser.parse_args()

    asset = find_asset(args.kind, args.name)
    hf_id = asset.get("hf_id")
    if not hf_id:
        raise SystemExit(f"{args.kind}:{args.name} has no hf_id in manifest")

    if args.kind == "model":
        local_dir = Path("models") / asset["name"]
        repo_type = "model"
        license_note = asset.get("license", "unknown")
    else:
        local_dir = Path(asset["raw_dir"])
        repo_type = asset.get("hf_type", "dataset")
        license_note = "requires license/access check" if asset.get("requires_license_check") else "check manifest"

    print(f"asset: {args.kind}:{args.name}")
    print(f"hf_id: {hf_id}")
    print(f"repo_type: {repo_type}")
    print(f"local_dir: {local_dir}")
    print(f"license/access: {license_note}")

    if args.dry_run:
        print("dry-run only; no download performed")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first: python -m pip install huggingface_hub") from exc

    local_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_download(
        repo_id=hf_id,
        repo_type=repo_type,
        local_dir=str(local_dir),
        revision=args.revision,
        allow_patterns=args.allow_pattern or None,
        ignore_patterns=args.ignore_pattern or None,
    )
    print(f"downloaded to {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
