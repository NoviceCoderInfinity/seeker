#!/usr/bin/env python3
"""Create a deterministic JSONL subset for kill tests."""

from __future__ import annotations

import argparse
import json
import random


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, required=True)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    rows = []
    with open(args.input, "r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    random.Random(args.seed).shuffle(rows)
    rows = rows[: args.limit]
    with open(args.output, "w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True))
            handle.write("\n")
    print(f"wrote {len(rows)} rows to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
