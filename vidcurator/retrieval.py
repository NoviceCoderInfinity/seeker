"""Frame retrieval and sampling primitives."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .types import EvidenceSpan, FrameRecord, QAExample

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> List[str]:
    return TOKEN_RE.findall(text.lower())


def token_set(text: str) -> Set[str]:
    return set(tokenize(text))


def lexical_score(query: str, text: str) -> float:
    q = token_set(query)
    if not q:
        return 0.0
    t = token_set(text)
    if not t:
        return 0.0
    overlap = len(q & t)
    return overlap / math.sqrt(len(q) * len(t))


def cosine(a: Optional[Sequence[float]], b: Optional[Sequence[float]]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(y * y for y in b))
    if na == 0.0 or nb == 0.0:
        return 0.0
    return dot / (na * nb)


def frame_text(frame: FrameRecord) -> str:
    extras = " ".join(str(v) for v in frame.metadata.values() if isinstance(v, (str, int, float)))
    return f"{frame.caption} {extras}".strip()


def normalize_video_id(video_id: str) -> str:
    """Map QA ids (e.g. ``clip.mp4``) and index ids (``clip``) to one lookup key."""
    return Path(video_id).stem


class FrameIndex:
    def __init__(self, frames: Iterable[FrameRecord]):
        self.by_video: Dict[str, List[FrameRecord]] = defaultdict(list)
        self.by_frame_id: Dict[str, FrameRecord] = {}
        for frame in frames:
            key = normalize_video_id(frame.video_id)
            self.by_video[key].append(frame)
            self.by_frame_id[f"{key}:{frame.frame_id}"] = frame
        for video_id in list(self.by_video):
            self.by_video[video_id].sort(key=lambda f: (f.timestamp, f.frame_id))

    def frames(self, video_id: str) -> List[FrameRecord]:
        return list(self.by_video.get(normalize_video_id(video_id), []))

    def get(self, video_id: str, frame_id: str) -> Optional[FrameRecord]:
        return self.by_frame_id.get(f"{normalize_video_id(video_id)}:{frame_id}")


class UniformSampler:
    def __init__(self, index: FrameIndex):
        self.index = index

    def sample(self, video_id: str, k: int) -> List[FrameRecord]:
        frames = self.index.frames(video_id)
        if k <= 0 or not frames:
            return []
        if k >= len(frames):
            return frames
        if k == 1:
            return [frames[len(frames) // 2]]
        step = (len(frames) - 1) / (k - 1)
        picks = []
        used = set()
        for i in range(k):
            idx = round(i * step)
            if idx not in used:
                picks.append(frames[idx])
                used.add(idx)
        return picks


class LexicalRetriever:
    def __init__(self, index: FrameIndex):
        self.index = index

    def retrieve(
        self,
        video_id: str,
        query: str,
        k: int,
        exclude_ids: Optional[Set[str]] = None,
    ) -> List[Tuple[FrameRecord, float]]:
        exclude_ids = exclude_ids or set()
        scored = []
        for frame in self.index.frames(video_id):
            if frame.frame_id in exclude_ids:
                continue
            score = lexical_score(query, frame_text(frame))
            scored.append((frame, score))
        scored.sort(key=lambda item: (item[1], -item[0].timestamp), reverse=True)
        return scored[:k]


class MMRRetriever:
    """Query relevance plus caption or embedding diversity.

    This is the dependency-free stand-in for the SigLIP/DINO relevance+coverage
    retrievers used in the real H100 runs. If frame embeddings are present, they
    are used for diversity; relevance still comes from captions/metadata unless
    a future policy supplies query-image/text embedding scores.
    """

    def __init__(self, index: FrameIndex, lambda_relevance: float = 0.72):
        self.index = index
        self.lambda_relevance = lambda_relevance

    def retrieve(
        self,
        video_id: str,
        query: str,
        k: int,
        exclude_ids: Optional[Set[str]] = None,
        query_embedding: Optional[Sequence[float]] = None,
    ) -> List[Tuple[FrameRecord, float]]:
        """MMR retrieval with optional SigLIP query-embedding relevance.

        When ``query_embedding`` is supplied (e.g. a SigLIP text embedding of the
        question) and a frame carries a SigLIP image ``embedding``, relevance is the
        cosine similarity in the shared SigLIP space — a faithful, label-free
        improvement over caption-lexical relevance (T1.b FEP fix). Falls back to
        lexical relevance per-frame when an embedding is missing.
        """
        exclude_ids = exclude_ids or set()
        candidates = [f for f in self.index.frames(video_id) if f.frame_id not in exclude_ids]
        selected: List[Tuple[FrameRecord, float]] = []
        selected_tokens: List[Set[str]] = []
        while candidates and len(selected) < k:
            best_idx = 0
            best_score = -1e9
            for idx, frame in enumerate(candidates):
                # Use SigLIP cosine relevance only when the query and frame
                # embeddings share the same space (equal length). On any mismatch
                # (different/embedding-less index) fall back to lexical relevance
                # rather than silently scoring everything 0.
                if (
                    query_embedding is not None
                    and frame.embedding
                    and len(query_embedding) == len(frame.embedding)
                ):
                    rel = cosine(query_embedding, frame.embedding)
                else:
                    rel = lexical_score(query, frame_text(frame))
                toks = token_set(frame_text(frame))
                redundancy = 0.0
                if selected and frame.embedding:
                    redundancy = max(cosine(frame.embedding, prior.embedding) for prior, _ in selected)
                elif selected_tokens and toks:
                    redundancy = max(len(toks & s) / max(1, len(toks | s)) for s in selected_tokens)
                score = self.lambda_relevance * rel - (1.0 - self.lambda_relevance) * redundancy
                if score > best_score:
                    best_idx = idx
                    best_score = score
            frame = candidates.pop(best_idx)
            selected.append((frame, max(0.0, best_score)))
            selected_tokens.append(token_set(frame_text(frame)))
        return selected


def query_embedding_for(example: QAExample) -> Optional[List[float]]:
    """Return a precomputed SigLIP query embedding from example metadata, if present.

    The SigLIP-so400m re-index pipeline (T1.b) stores a text embedding of the
    question (optionally fused with options) under ``metadata['query_embedding']``.
    This is label-free — it never reads gold evidence spans — so it is safe for
    deployable, controlled-comparison policies. Returns None when unavailable, in
    which case retrievers fall back to caption-lexical relevance.
    """
    if not example.metadata:
        return None
    raw = example.metadata.get("query_embedding")
    if raw is None:
        return None
    try:
        return [float(x) for x in raw]
    except (TypeError, ValueError):
        return None


def timestamp_hits_spans(timestamp: float, spans: Sequence[EvidenceSpan], tolerance: float = 0.0) -> bool:
    return any(span.contains(timestamp, tolerance=tolerance) for span in spans)


def retrieval_recall_at_k(index: FrameIndex, examples: Sequence[QAExample], k: int) -> float:
    retriever = MMRRetriever(index)
    total = 0
    hits = 0
    for example in examples:
        if not example.evidence_spans:
            continue
        total += 1
        query = " ".join([example.question] + example.options)
        frames = [frame for frame, _ in retriever.retrieve(example.video_id, query, k)]
        if any(timestamp_hits_spans(frame.timestamp, example.evidence_spans) for frame in frames):
            hits += 1
    return hits / total if total else 0.0
