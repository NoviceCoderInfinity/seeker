"""Command line entry point."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .adapters import normalize_dataset
from .evaluate import evaluate_config
from .io import read_json


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="vidcurator", description="VID-CURATOR experiment tools")
    subparsers = parser.add_subparsers(dest="command", required=True)

    eval_parser = subparsers.add_parser("evaluate", help="run policies and write predictions/summary")
    eval_parser.add_argument("--config", required=True, help="JSON config path")

    norm_parser = subparsers.add_parser("normalize", help="normalize benchmark QA files to project JSONL")
    norm_parser.add_argument("--dataset", required=True)
    norm_parser.add_argument("--input", required=True)
    norm_parser.add_argument("--output", required=True)

    args = parser.parse_args(argv)
    if args.command == "evaluate":
        summary = evaluate_config(read_json(args.config))
        print(f"wrote {Path(summary['config'].get('output_dir', 'runs/current')).resolve()}")
        for key, value in sorted(summary["groups"].items()):
            acc = value["accuracy"]
            frames = value["avg_frames"]
            print(f"{key}: acc={acc:.3f} avg_frames={frames:.2f}")
        return 0
    if args.command == "normalize":
        normalize_dataset(args.dataset, args.input, args.output)
        print(f"wrote {Path(args.output).resolve()}")
        return 0
    return 2


if __name__ == "__main__":
    sys.exit(main())
