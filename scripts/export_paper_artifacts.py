#!/usr/bin/env python3
"""Export CSV/SVG paper artifacts from a run directory."""

from __future__ import annotations

import argparse

from vidcurator.artifacts import export_run_artifacts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    args = parser.parse_args()
    outputs = export_run_artifacts(args.run_dir)
    for key, value in sorted(outputs.items()):
        print(f"{key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
