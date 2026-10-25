"""Qualitative trace rendering for paper appendices and debugging."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List

from .io import ensure_dir, load_examples, load_frames, read_jsonl
from .types import QAExample


def render_qualitative_traces(
    run_dir: str | Path,
    qa_path: str,
    frames_path: str,
    max_examples: int = 12,
    policies: Iterable[str] | None = None,
) -> List[str]:
    run_path = Path(run_dir)
    out_dir = run_path / "qualitative_traces"
    ensure_dir(out_dir)
    predictions = read_jsonl(run_path / "predictions.jsonl")
    examples = {example.qid: example for example in load_examples(qa_path)}
    frame_lookup = {
        (frame.video_id, frame.frame_id): frame
        for frame in load_frames(frames_path)
    }
    policy_filter = set(policies or [])
    selected = []
    for row in predictions:
        if policy_filter and row["policy"] not in policy_filter:
            continue
        selected.append(row)
        if len(selected) >= max_examples:
            break

    written = []
    for idx, row in enumerate(selected, start=1):
        example = examples[row["qid"]]
        path = out_dir / f"{idx:03d}_{row['policy']}_{row['qid']}.md"
        path.write_text(_render_trace(row, example, frame_lookup), encoding="utf-8")
        written.append(str(path))
    return written


def _render_trace(row: Dict[str, Any], example: QAExample, frame_lookup: Dict[tuple, Any]) -> str:
    lines = [
        f"# {row['policy']} / {row['qid']}",
        "",
        f"**Video:** `{row['video_id']}`",
        f"**Budget:** `{row['budget']}`",
        f"**Question:** {example.question}",
        "",
    ]
    if example.options:
        lines.append("**Options:**")
        for idx, option in enumerate(example.options):
            marker = " (gold)" if example.answer_index == idx else ""
            lines.append(f"- `{idx}` {option}{marker}")
        lines.append("")
    lines.extend(
        [
            f"**Prediction:** `{row.get('answer_index')}` {row.get('answer')}",
            f"**Confidence:** `{row.get('confidence'):.3f}`",
            "",
            "## Selected Frames",
            "",
        ]
    )
    for frame_id, timestamp in zip(row.get("selected_frame_ids", []), row.get("selected_timestamps", [])):
        frame = frame_lookup.get((row["video_id"], frame_id))
        caption = frame.caption if frame else ""
        path = frame.path if frame else ""
        lines.append(f"- `{frame_id}` at `{timestamp:.1f}s`: {caption}")
        if path:
            lines.append(f"  - path: `{path}`")
    if example.evidence_spans:
        lines.extend(["", "## Gold Evidence Spans", ""])
        for span in example.evidence_spans:
            lines.append(f"- `{span.start:.1f}s` to `{span.end:.1f}s`: {span.label}")
    if row.get("probes"):
        lines.extend(["", "## Probes", ""])
        for probe in row["probes"]:
            lines.append(f"- sufficiency `{probe.get('sufficiency', 0):.3f}`; next `{probe.get('next_action')}`")
            lines.append(f"  - missing: {probe.get('missing_moment', '')}")
    if row.get("trace"):
        lines.extend(["", "## Event Trace", ""])
        for event in row["trace"]:
            lines.append(f"- `{event.get('event')}`")
            if "query" in event:
                lines.append(f"  - query: {event['query']}")
            if "frames" in event:
                frame_bits = ", ".join(f"{x.get('frame_id')}:{x.get('score', 0):.3f}" for x in event["frames"])
                lines.append(f"  - frames: {frame_bits}")
            if "answer" in event:
                lines.append(f"  - tentative: `{event.get('answer_index')}` {event.get('answer')}")
    lines.append("")
    return "\n".join(lines)
