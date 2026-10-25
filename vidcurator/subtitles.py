"""Subtitle/ASR ingestion as a ledger text stream (VID-MEM pilot P1).

Benchmarks ship transcripts (videomme: .srt per YouTube id; longvideobench: {vid}_en.json
with start/end/line). We treat them as another timestamped episodic-memory stream — same
format as DdE scan captions — NOT as a prompt hack: entries are dated, interleavable with
captions, and truncated to a character budget by even temporal downsampling (never head-cut,
which would drop the whole second half of the video).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional, Tuple

_TAG_RE = re.compile(r"<[^>]+>")
_TS_RE = re.compile(r"(\d+):(\d+):(\d+)[,.](\d+)")


def _ts_to_seconds(ts: str) -> Optional[float]:
    m = _TS_RE.search(ts)
    if not m:
        return None
    h, mi, s, ms = m.groups()
    return int(h) * 3600 + int(mi) * 60 + int(s) + int(ms[:3].ljust(3, "0")) / 1000.0


def _parse_srt(text: str) -> List[Tuple[float, str]]:
    out: List[Tuple[float, str]] = []
    for block in re.split(r"\n\s*\n", text):
        lines = [ln.strip() for ln in block.strip().splitlines()]
        if len(lines) < 2 or "-->" not in "".join(lines[:2]):
            continue
        ts_line = lines[1] if "-->" in lines[1] else lines[0]
        start = _ts_to_seconds(ts_line.split("-->")[0])
        if start is None:
            continue
        content = " ".join(_TAG_RE.sub("", ln) for ln in lines[2:] if "-->" not in ln).strip()
        if content:
            out.append((start, content))
    return out


def _parse_lvb_json(path: Path) -> List[Tuple[float, str]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out: List[Tuple[float, str]] = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        start = _ts_to_seconds(str(r.get("start", "")))
        line = str(r.get("line", "")).strip()
        if start is not None and line:
            out.append((start, line))
    return out


def load_subtitles(video_id: str, subtitle_dir: str) -> List[Tuple[float, str]]:
    """Return [(start_seconds, text)] for a video, or [] if no transcript exists."""
    root = Path(subtitle_dir)
    candidates = [
        root / f"{video_id}.srt",
        root / "subtitle" / f"{video_id}.srt",
        root / f"{video_id}_en.json",
        root / "subtitles" / f"{video_id}_en.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        try:
            if path.suffix == ".srt":
                return _parse_srt(path.read_text(encoding="utf-8", errors="replace"))
            return _parse_lvb_json(path)
        except (OSError, ValueError, json.JSONDecodeError):
            return []
    return []


def _render(entries: List[Tuple[float, str]]) -> str:
    return "\n".join(f"[t={int(t)}s sub] {txt}" for t, txt in entries)


def downsample_entries(subs: List[Tuple[float, str]], max_chars: int = 4000) -> List[Tuple[float, str]]:
    """Evenly downsample in TIME (drop every other entry) until the rendered text fits."""
    entries = list(subs)
    while len(_render(entries)) > max_chars and len(entries) > 1:
        entries = entries[::2]
    return entries


def render_subtitle_block(subs: List[Tuple[float, str]], max_chars: int = 4000) -> str:
    """Timestamped transcript lines, evenly downsampled in TIME to fit max_chars."""
    if not subs:
        return ""
    return _render(downsample_entries(subs, max_chars))[:max_chars]
