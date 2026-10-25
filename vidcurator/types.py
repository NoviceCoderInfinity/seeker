"""Core data contracts for VID-CURATOR.

The project uses JSONL artifacts at the boundaries so benchmark adapters,
offline frame indexing, model servers, and RL training can evolve independently.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence


def _float_list(value: Any) -> Optional[List[float]]:
    if value is None:
        return None
    return [float(x) for x in value]


@dataclass(frozen=True)
class EvidenceSpan:
    start: float
    end: float
    label: str = ""

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "EvidenceSpan":
        return cls(
            start=float(raw.get("start", raw.get("start_time", 0.0))),
            end=float(raw.get("end", raw.get("end_time", 0.0))),
            label=str(raw.get("label", "")),
        )

    def contains(self, timestamp: float, tolerance: float = 0.0) -> bool:
        return self.start - tolerance <= timestamp <= self.end + tolerance


@dataclass(frozen=True)
class FrameRecord:
    video_id: str
    frame_id: str
    timestamp: float
    path: str = ""
    caption: str = ""
    embedding: Optional[List[float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "FrameRecord":
        return cls(
            video_id=str(raw["video_id"]),
            frame_id=str(raw.get("frame_id", raw.get("id", ""))),
            timestamp=float(raw.get("timestamp", raw.get("time", 0.0))),
            path=str(raw.get("path", raw.get("image", ""))),
            caption=str(raw.get("caption", raw.get("summary", ""))),
            embedding=_float_list(raw.get("embedding")),
            metadata=dict(raw.get("metadata", {})),
        )

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class QAExample:
    qid: str
    video_id: str
    question: str
    options: List[str] = field(default_factory=list)
    answer: str = ""
    answer_index: Optional[int] = None
    evidence_spans: List[EvidenceSpan] = field(default_factory=list)
    domain: str = "unknown"
    split: str = "eval"
    metadata: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "QAExample":
        spans = raw.get("evidence_spans", raw.get("temporal_spans", [])) or []
        answer_index = raw.get("answer_index", raw.get("answer_idx"))
        return cls(
            qid=str(raw.get("qid", raw.get("id", ""))),
            video_id=str(raw["video_id"]),
            question=str(raw["question"]),
            options=[str(x) for x in raw.get("options", [])],
            answer=str(raw.get("answer", "")),
            answer_index=int(answer_index) if answer_index is not None else None,
            evidence_spans=[EvidenceSpan.from_dict(x) for x in spans],
            domain=str(raw.get("domain", "unknown")),
            split=str(raw.get("split", "eval")),
            metadata=dict(raw.get("metadata", {})),
        )

    def correct_answer_text(self) -> str:
        if self.answer:
            return self.answer
        if self.answer_index is not None and 0 <= self.answer_index < len(self.options):
            return self.options[self.answer_index]
        return ""

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        raw["evidence_spans"] = [asdict(span) for span in self.evidence_spans]
        return raw


@dataclass(frozen=True)
class ContextProbe:
    held_moments: List[str]
    missing_moment: str
    sufficiency: float
    next_action: str
    predicted_window: Optional[EvidenceSpan] = None
    rationale: str = ""

    def to_dict(self) -> Dict[str, Any]:
        raw = asdict(self)
        if self.predicted_window is not None:
            raw["predicted_window"] = asdict(self.predicted_window)
        return raw


@dataclass(frozen=True)
class BufferBudget:
    max_frames: int = 8
    max_summary_chars: int = 2000


@dataclass
class BufferFrame:
    frame: FrameRecord
    usefulness: float
    hop_added: int
    note: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "frame": self.frame.to_dict(),
            "usefulness": self.usefulness,
            "hop_added": self.hop_added,
            "note": self.note,
        }


@dataclass
class BufferState:
    question: str
    options: Sequence[str]
    budget: BufferBudget
    keyframes: List[BufferFrame] = field(default_factory=list)
    running_summary: str = ""
    probes: List[ContextProbe] = field(default_factory=list)
    open_gaps: List[str] = field(default_factory=list)
    resolved: bool = False


@dataclass(frozen=True)
class AnswerPrediction:
    answer: str
    answer_index: Optional[int]
    confidence: float
    raw: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyResult:
    qid: str
    video_id: str
    policy: str
    budget: int
    answer: str
    answer_index: Optional[int]
    confidence: float
    selected_frame_ids: List[str]
    selected_timestamps: List[float]
    probes: List[Dict[str, Any]]
    num_retrievals: int
    num_frames: int
    trace: List[Dict[str, Any]]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "PolicyResult":
        return cls(
            qid=str(raw["qid"]),
            video_id=str(raw["video_id"]),
            policy=str(raw["policy"]),
            budget=int(raw["budget"]),
            answer=str(raw.get("answer", "")),
            answer_index=raw.get("answer_index"),
            confidence=float(raw.get("confidence", 0.0)),
            selected_frame_ids=list(raw.get("selected_frame_ids", [])),
            selected_timestamps=[float(t) for t in raw.get("selected_timestamps", [])],
            probes=list(raw.get("probes", [])),
            num_retrievals=int(raw.get("num_retrievals", 0)),
            num_frames=int(raw.get("num_frames", 0)),
            trace=list(raw.get("trace", [])),
            metadata=dict(raw.get("metadata", {})),
        )
