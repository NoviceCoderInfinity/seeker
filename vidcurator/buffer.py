"""Running-context buffer implementation."""

from __future__ import annotations

from typing import Iterable, List, Sequence

from .types import BufferBudget, BufferFrame, BufferState, ContextProbe, FrameRecord


class RunningContextBuffer:
    def __init__(self, question: str, options: Sequence[str], budget: BufferBudget):
        self.state = BufferState(question=question, options=options, budget=budget)

    def add_probe(self, probe: ContextProbe) -> None:
        self.state.probes.append(probe)
        self.state.open_gaps = [probe.missing_moment] if probe.missing_moment else []

    def add_frame(self, frame: FrameRecord, usefulness: float, hop: int, note: str = "") -> None:
        if frame.frame_id in {item.frame.frame_id for item in self.state.keyframes}:
            return
        self.state.keyframes.append(BufferFrame(frame=frame, usefulness=usefulness, hop_added=hop, note=note))
        self.state.keyframes.sort(key=lambda item: (-item.usefulness, item.frame.timestamp))
        self._enforce_budget()

    def selected_ids(self) -> List[str]:
        return [item.frame.frame_id for item in self.state.keyframes]

    def selected_timestamps(self) -> List[float]:
        return [item.frame.timestamp for item in self.state.keyframes]

    def frames(self) -> List[FrameRecord]:
        return [item.frame for item in sorted(self.state.keyframes, key=lambda item: item.frame.timestamp)]

    def probes(self) -> List[ContextProbe]:
        return list(self.state.probes)

    def _enforce_budget(self) -> None:
        budget = self.state.budget
        while len(self.state.keyframes) > budget.max_frames:
            dropped = self.state.keyframes.pop()
            self._compress_frame(dropped)
        if len(self.state.running_summary) > budget.max_summary_chars:
            self.state.running_summary = self.state.running_summary[-budget.max_summary_chars :]

    def compress_frame(self, frame_id: str) -> bool:
        """Force-COMPRESS a kept frame: drop its pixels, keep its caption in the running summary.

        Used by the ForcedCompress ablation (E-K) to textualize evidence frames without dropping
        the information they carry — isolating 'what survives textualization'.
        """
        for i, item in enumerate(self.state.keyframes):
            if item.frame.frame_id == frame_id:
                self.state.keyframes.pop(i)
                self._compress_frame(item)
                return True
        return False

    def _compress_frame(self, item: BufferFrame) -> None:
        caption = item.frame.caption or item.note or "uncaptioned frame"
        entry = f"[t={item.frame.timestamp:.1f}s dropped:{item.frame.frame_id}] {caption}"
        if self.state.running_summary:
            self.state.running_summary += "\n"
        self.state.running_summary += entry

    def render_text(self) -> str:
        lines = [f"Question: {self.state.question}"]
        if self.state.options:
            lines.append("Options:")
            for idx, option in enumerate(self.state.options):
                letter = chr(ord("A") + idx) if idx < 26 else str(idx)
                lines.append(f"{letter}. {option}")
        if self.state.running_summary:
            lines.append("Running summary:")
            lines.append(self.state.running_summary)
        lines.append("Keyframes:")
        for item in sorted(self.state.keyframes, key=lambda x: x.frame.timestamp):
            caption = item.frame.caption or item.note
            lines.append(
                f"- id={item.frame.frame_id} t={item.frame.timestamp:.1f}s "
                f"use={item.usefulness:.3f}: {caption}"
            )
        if self.state.probes:
            last = self.state.probes[-1]
            lines.append(
                f"Last probe: sufficiency={last.sufficiency:.2f}; "
                f"missing={last.missing_moment}; next={last.next_action}"
            )
        return "\n".join(lines)

    @staticmethod
    def from_frames(
        question: str,
        options: Sequence[str],
        frames: Iterable[FrameRecord],
        budget: BufferBudget,
    ) -> "RunningContextBuffer":
        buffer = RunningContextBuffer(question=question, options=options, budget=budget)
        for frame in frames:
            buffer.add_frame(frame, usefulness=1.0, hop=0)
        return buffer
