#!/usr/bin/env python3
"""Render qualitative trace markdown files from a completed run."""

from __future__ import annotations

import argparse

from vidcurator.traces import render_qualitative_traces


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--qa-path", required=True)
    parser.add_argument("--frames-path", required=True)
    parser.add_argument("--max-examples", type=int, default=12)
    parser.add_argument("--policy", action="append", default=[])
    args = parser.parse_args()
    paths = render_qualitative_traces(
        args.run_dir,
        args.qa_path,
        args.frames_path,
        max_examples=args.max_examples,
        policies=args.policy or None,
    )
    for path in paths:
        print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
