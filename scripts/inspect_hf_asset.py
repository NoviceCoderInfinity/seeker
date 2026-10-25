#!/usr/bin/env python3
"""Inspect Hugging Face repo files for a manifest asset."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from download_hf_assets import find_asset


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--kind", choices=["model", "dataset"], required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--limit", type=int, default=80)
    parser.add_argument("--output", default="")
    args = parser.parse_args()

    try:
        from huggingface_hub import HfApi
    except ImportError as exc:
        raise SystemExit("Install huggingface_hub first") from exc

    asset = find_asset(args.kind, args.name)
    hf_id = asset["hf_id"]
    repo_type = "model" if args.kind == "model" else asset.get("hf_type", "dataset")
    api = HfApi()
    info = api.repo_info(repo_id=hf_id, repo_type=repo_type, files_metadata=True)
    rows = []
    for sibling in info.siblings:
        rows.append(
            {
                "path": sibling.rfilename,
                "size": getattr(sibling, "size", None),
                "blob_id": getattr(sibling, "blob_id", None),
                "lfs": bool(getattr(sibling, "lfs", None)),
            }
        )
    total_size = sum(row["size"] or 0 for row in rows)
    payload = {
        "kind": args.kind,
        "name": args.name,
        "hf_id": hf_id,
        "repo_type": repo_type,
        "num_files": len(rows),
        "known_total_size_bytes": total_size,
        "files": rows,
    }
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"{args.kind}:{args.name} -> {hf_id}")
    print(f"files={len(rows)} known_total_size_bytes={total_size}")
    for row in rows[: args.limit]:
        size = row["size"]
        size_text = "unknown" if size is None else f"{size / (1024 ** 2):.2f} MiB"
        print(f"{size_text}\t{row['path']}")
    if len(rows) > args.limit:
        print(f"... {len(rows) - args.limit} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
