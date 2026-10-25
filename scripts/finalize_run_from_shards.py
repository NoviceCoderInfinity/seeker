#!/usr/bin/env python3
"""Rebuild run summary.json from completed _shards/*.jsonl files."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from vidcurator.evaluate import expected_shard_count, finalize_from_shards


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Full run config (all policies/budgets)")
    parser.add_argument("--force", action="store_true", help="Overwrite existing summary.json")
    args = parser.parse_args()

    cfg = json.loads(Path(args.config).read_text())
    out = Path(cfg["output_dir"])
    if (out / "summary.json").exists() and not args.force:
        print(f"skip {out}/summary.json (exists; use --force)")
        return 0

    budgets = [int(x) for x in cfg.get("budgets", [])]
    policies = list(cfg.get("policies", []))
    need = expected_shard_count(budgets, policies)
    have = len(list((out / "_shards").glob("*.jsonl"))) if (out / "_shards").exists() else 0
    if have < need:
        print(f"not ready {out}: {have}/{need} shards", file=sys.stderr)
        return 1

    summary = finalize_from_shards(cfg)
    print(f"wrote {out}/summary.json groups={len(summary.get('groups', {}))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
