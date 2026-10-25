"""Curator and baseline policies."""

from __future__ import annotations

import heapq

from typing import Any, Dict, List, Optional, Sequence, Set

from .answerers import BaseAnswerer
from .buffer import RunningContextBuffer
from .retrieval import (
    FrameIndex,
    LexicalRetriever,
    MMRRetriever,
    UniformSampler,
    frame_text,
    lexical_score,
    query_embedding_for,
    timestamp_hits_spans,
)
from .types import AnswerPrediction, BufferBudget, ContextProbe, FrameRecord, PolicyResult, QAExample


class BasePolicy:
    name = "base"

    def run(self, example: QAExample) -> PolicyResult:
        raise NotImplementedError


def _subtitle_block(example: QAExample, subtitle_dir: Optional[str], max_chars: int) -> str:
    """Transcript stream for the ledger; '' when disabled or missing (vision-only fallback)."""
    if not subtitle_dir:
        return ""
    from .subtitles import load_subtitles, render_subtitle_block

    return render_subtitle_block(load_subtitles(example.video_id, subtitle_dir), max_chars)


class UniformPolicy(BasePolicy):
    name = "uniform"

    def __init__(
        self,
        index: FrameIndex,
        answerer: BaseAnswerer,
        budget: int,
        subtitle_dir: Optional[str] = None,
        subtitle_chars: int = 4000,
        subtitle_mode: str = "block",
    ):
        self.index = index
        self.answerer = answerer
        self.budget = budget
        self.sampler = UniformSampler(index)
        self.subtitle_dir = subtitle_dir
        self.subtitle_chars = subtitle_chars
        # "block": transcript prepended as one text block. "interleaved": subtitle lines
        # placed between the frame images at their timestamps (multi-stream fusion pilot).
        self.subtitle_mode = subtitle_mode

    def run(self, example: QAExample) -> PolicyResult:
        frames = self.sampler.sample(example.video_id, self.budget)
        buffer = RunningContextBuffer.from_frames(
            example.question,
            example.options,
            frames,
            BufferBudget(max_frames=self.budget),
        )
        context = buffer.render_text()
        if self.subtitle_mode == "interleaved" and self.subtitle_dir:
            from .subtitles import downsample_entries, load_subtitles

            events = downsample_entries(
                load_subtitles(example.video_id, self.subtitle_dir), self.subtitle_chars
            )
            pred = self.answerer.answer(example, frames, context, subtitle_events=events or None)
            trace = [{"event": "uniform_sample", "subtitle_mode": "interleaved",
                      "subtitle_events": len(events)}]
            return make_result(example, self.name, self.budget, pred, frames, [], 0, trace)
        subs = _subtitle_block(example, self.subtitle_dir, self.subtitle_chars)
        if subs:
            context = f"Transcript:\n{subs}\n\n{context}"
        pred = self.answerer.answer(example, frames, context)
        trace = [{"event": "uniform_sample", "subtitle_chars": len(subs)}]
        return make_result(example, self.name, self.budget, pred, frames, [], 0, trace)


class BlindPolicy(BasePolicy):
    """Answer with NO frames — measures the answerer's language prior / question leakage.
    Control for the learned curator's global-task behavior: if learned≈blind on a dataset,
    the "win" is prior-driven (curation-avoidance), not better frame selection."""
    name = "blind"

    def __init__(self, index: FrameIndex, answerer: BaseAnswerer, budget: int):
        self.index = index
        self.answerer = answerer
        self.budget = budget

    def run(self, example: QAExample) -> PolicyResult:
        frames: list = []
        buffer = RunningContextBuffer.from_frames(
            example.question,
            example.options,
            frames,
            BufferBudget(max_frames=self.budget),
        )
        pred = self.answerer.answer(example, frames, buffer.render_text())
        return make_result(example, self.name, self.budget, pred, frames, [], 0, [{"event": "blind_no_frames"}])


class MMRPolicy(BasePolicy):
    name = "mmr_retrieve"

    def __init__(self, index: FrameIndex, answerer: BaseAnswerer, budget: int):
        self.index = index
        self.answerer = answerer
        self.budget = budget
        self.retriever = MMRRetriever(index)

    def run(self, example: QAExample) -> PolicyResult:
        query = " ".join([example.question] + example.options)
        frames = [frame for frame, _ in self.retriever.retrieve(example.video_id, query, self.budget)]
        buffer = RunningContextBuffer.from_frames(
            example.question,
            example.options,
            frames,
            BufferBudget(max_frames=self.budget),
        )
        pred = self.answerer.answer(example, frames, buffer.render_text())
        return make_result(example, self.name, self.budget, pred, frames, [], 1, [{"event": "mmr_retrieve"}])


class BestSingleFramePolicy(BasePolicy):
    name = "best_single_frame"

    def __init__(self, index: FrameIndex, answerer: BaseAnswerer, budget: int):
        self.index = index
        self.answerer = answerer
        self.budget = budget

    def run(self, example: QAExample) -> PolicyResult:
        best_pred = None
        best_frame = None
        trace: List[Dict[str, Any]] = []
        pool = self.index.frames(example.video_id)
        # Best-single-frame is a budget-independent audit baseline. Scanning the full pool
        # is intractable on long-video sets (videomme median 243, mlvu up to 821 frames/video).
        # Score a bounded uniform subsample of candidates: "best single frame among N samples".
        import os
        cap = int(os.environ.get("SINGLE_FRAME_MAX_CANDIDATES", "32"))
        if cap > 0 and len(pool) > cap:
            step = len(pool) / cap
            pool = [pool[int(i * step)] for i in range(cap)]
        for frame in pool:
            buffer = RunningContextBuffer.from_frames(
                example.question,
                example.options,
                [frame],
                BufferBudget(max_frames=1),
            )
            pred = self.answerer.answer(example, [frame], buffer.render_text())
            trace.append({"event": "single_frame_score", "frame_id": frame.frame_id, "confidence": pred.confidence})
            if best_pred is None or pred.confidence > best_pred.confidence:
                best_pred = pred
                best_frame = frame
        frames = [best_frame] if best_frame is not None else []
        if best_pred is None:
            best_pred = self.answerer.answer(
                example,
                [],
                RunningContextBuffer(example.question, example.options, BufferBudget(max_frames=0)).render_text(),
            )
        return make_result(example, self.name, 1, best_pred, frames, [], len(self.index.frames(example.video_id)), trace)


class OracleSpanPolicy(BasePolicy):
    """Evidence-span upper bound for grounded datasets.

    This is not a deployable method. It tells us whether the model can answer
    when it receives frames from annotated gold moments.
    """

    name = "oracle_span"

    def __init__(self, index: FrameIndex, answerer: BaseAnswerer, budget: int):
        self.index = index
        self.answerer = answerer
        self.budget = budget
        self.fallback = MMRRetriever(index)

    def run(self, example: QAExample) -> PolicyResult:
        trace: List[Dict[str, Any]] = []
        frames = [
            frame
            for frame in self.index.frames(example.video_id)
            if timestamp_hits_spans(frame.timestamp, example.evidence_spans, tolerance=1.0)
        ][: self.budget]
        if len(frames) < self.budget:
            exclude = {frame.frame_id for frame in frames}
            query = " ".join([example.question] + example.options)
            frames.extend(
                frame
                for frame, _ in self.fallback.retrieve(
                    example.video_id,
                    query,
                    self.budget - len(frames),
                    exclude_ids=exclude,
                )
            )
        trace.append({"event": "oracle_span_select", "has_spans": bool(example.evidence_spans)})
        buffer = RunningContextBuffer.from_frames(
            example.question,
            example.options,
            frames,
            BufferBudget(max_frames=self.budget),
        )
        pred = self.answerer.answer(example, frames, buffer.render_text())
        return make_result(example, self.name, self.budget, pred, frames, [], 1, trace)


class OracleSpanContextPolicy(BasePolicy):
    """Corrected evidence upper bound: spread span frames + uniform temporal context.

    ``oracle_span`` takes the first ``budget`` in-span frames — consecutive near-duplicates —
    and in practice scores BELOW the curator (X3). This variant fixes both defects:
    half the budget is in-span frames subsampled evenly across the span(s) (no duplicates),
    half is uniform frames over the whole video (the context the answerer also needs).
    Still not deployable (reads gold spans); headline-banned like oracle_span.
    """

    name = "oracle_span_context"

    def __init__(self, index: FrameIndex, answerer: BaseAnswerer, budget: int):
        self.index = index
        self.answerer = answerer
        self.budget = budget
        self.uniform = UniformSampler(index)

    @staticmethod
    def _spread(items: List[FrameRecord], k: int) -> List[FrameRecord]:
        if len(items) <= k:
            return list(items)
        step = len(items) / k
        return [items[int(i * step)] for i in range(k)]

    def run(self, example: QAExample) -> PolicyResult:
        trace: List[Dict[str, Any]] = []
        span_frames = [
            frame
            for frame in self.index.frames(example.video_id)
            if timestamp_hits_spans(frame.timestamp, example.evidence_spans, tolerance=1.0)
        ]
        k_span = min(len(span_frames), max(1, self.budget // 2)) if span_frames else 0
        frames = self._spread(span_frames, k_span)
        chosen = {frame.frame_id for frame in frames}
        for frame in self.uniform.sample(example.video_id, self.budget):
            if len(frames) >= self.budget:
                break
            if frame.frame_id not in chosen:
                frames.append(frame)
                chosen.add(frame.frame_id)
        frames.sort(key=lambda fr: fr.timestamp)
        trace.append({"event": "oracle_span_context_select", "n_span": k_span,
                      "has_spans": bool(example.evidence_spans)})
        buffer = RunningContextBuffer.from_frames(
            example.question,
            example.options,
            frames,
            BufferBudget(max_frames=self.budget),
        )
        pred = self.answerer.answer(example, frames, buffer.render_text())
        return make_result(example, self.name, self.budget, pred, frames, [], 1, trace)


class CuratorProbeOffPolicy(BasePolicy):
    """Ablation: same retrieval/answer surface, no missing-moment probe."""

    name = "curator_probe_off"

    def __init__(self, index: FrameIndex, answerer: BaseAnswerer, budget: int, retrieve_per_hop: int = 4):
        self.index = index
        self.answerer = answerer
        self.budget = budget
        self.retrieve_per_hop = retrieve_per_hop
        self.retriever = MMRRetriever(index)

    def run(self, example: QAExample) -> PolicyResult:
        query = " ".join([example.question] + example.options)
        retrieved = self.retriever.retrieve(example.video_id, query, min(self.budget, self.retrieve_per_hop))
        frames = [frame for frame, _ in retrieved]
        buffer = RunningContextBuffer.from_frames(
            example.question,
            example.options,
            frames,
            BufferBudget(max_frames=self.budget),
        )
        pred = self.answerer.answer(example, frames, buffer.render_text())
        trace = [
            {
                "event": "probe_off_retrieve",
                "query": query,
                "frames": [{"frame_id": f.frame_id, "score": s} for f, s in retrieved],
            }
        ]
        return make_result(example, self.name, self.budget, pred, frames, [], 1, trace)


class CuratorCurationOffPolicy(BasePolicy):
    """Ablation: probe/retrieve loop without keep/compress/drop scoring."""

    name = "curator_curation_off"

    def __init__(
        self,
        index: FrameIndex,
        answerer: BaseAnswerer,
        budget: int,
        max_hops: int = 3,
        retrieve_per_hop: int = 4,
    ):
        self.index = index
        self.answerer = answerer
        self.budget = budget
        self.max_hops = max_hops
        self.retrieve_per_hop = retrieve_per_hop
        self.retriever = MMRRetriever(index)

    def run(self, example: QAExample) -> PolicyResult:
        selected: List[FrameRecord] = []
        trace: List[Dict[str, Any]] = []
        query = " ".join([example.question] + example.options)
        for hop in range(1, self.max_hops + 1):
            if len(selected) >= self.budget:
                break
            exclude = {frame.frame_id for frame in selected}
            retrieved = self.retriever.retrieve(
                example.video_id,
                query,
                min(self.retrieve_per_hop, self.budget - len(selected)),
                exclude_ids=exclude,
            )
            selected.extend(frame for frame, _ in retrieved)
            trace.append(
                {
                    "event": "curation_off_retrieve",
                    "hop": hop,
                    "query": query,
                    "frames": [{"frame_id": f.frame_id, "score": s} for f, s in retrieved],
                }
            )
            query = f"{query} " + " ".join(frame.caption for frame, _ in retrieved)
        buffer = RunningContextBuffer.from_frames(
            example.question,
            example.options,
            selected[: self.budget],
            BufferBudget(max_frames=self.budget),
        )
        pred = self.answerer.answer(example, buffer.frames(), buffer.render_text())
        return make_result(example, self.name, self.budget, pred, buffer.frames(), [], self.max_hops, trace)


class ScriptedCuratorPolicy(BasePolicy):
    name = "scripted_curator"

    def __init__(
        self,
        index: FrameIndex,
        answerer: BaseAnswerer,
        budget: int,
        max_hops: int = 3,
        retrieve_per_hop: int = 4,
        seed_uniform: int = 2,
        sufficiency_threshold: float = 0.68,
        enable_summary: bool = True,
        subtitle_dir: Optional[str] = None,
        subtitle_chars: int = 4000,
    ):
        self.index = index
        self.answerer = answerer
        self.budget = budget
        self.max_hops = max_hops
        self.retrieve_per_hop = retrieve_per_hop
        self.seed_uniform = seed_uniform
        self.sufficiency_threshold = sufficiency_threshold
        self.enable_summary = enable_summary
        self.subtitle_dir = subtitle_dir
        self.subtitle_chars = subtitle_chars
        self.sampler = UniformSampler(index)
        self.retriever = MMRRetriever(index)
        self.lexical = LexicalRetriever(index)

    def _inject_subtitles(self, example: QAExample, buffer: RunningContextBuffer) -> None:
        subs = _subtitle_block(example, self.subtitle_dir, self.subtitle_chars)
        if not subs:
            return
        needed = len(subs) + 2000  # transcript + headroom for probe/ledger summaries
        if buffer.state.budget.max_summary_chars < needed:
            buffer.state.budget = BufferBudget(
                max_frames=buffer.state.budget.max_frames,
                max_summary_chars=needed,
            )
        buffer.state.running_summary = (
            f"{subs}\n{buffer.state.running_summary}" if buffer.state.running_summary else subs
        )

    def run(self, example: QAExample) -> PolicyResult:
        buffer = RunningContextBuffer(
            example.question,
            example.options,
            BufferBudget(max_frames=self.budget),
        )
        trace: List[Dict[str, Any]] = []
        for frame in self.sampler.sample(example.video_id, min(self.seed_uniform, self.budget)):
            buffer.add_frame(frame, usefulness=0.25, hop=0, note="coarse thumbnail prior")
        self._inject_subtitles(example, buffer)
        self._pre_loop(example, buffer, trace)
        num_retrievals = 0
        pred = None

        for hop in range(1, self.max_hops + 1):
            probe = self._probe(example, buffer, hop)
            buffer.add_probe(probe)
            trace.append({"event": "probe", "hop": hop, "probe": probe.to_dict()})
            if probe.next_action == "answer":
                break

            query = self._query(example, probe, buffer)
            exclude: Set[str] = set(buffer.selected_ids())
            retrieved = self._retrieve(example, query, self.retrieve_per_hop, exclude)
            num_retrievals += 1
            for frame, score in retrieved:
                usefulness = max(score, lexical_score(example.question, frame_text(frame)))
                buffer.add_frame(frame, usefulness=usefulness, hop=hop, note=f"query={query}")
            self._post_retrieve(buffer, retrieved, hop)
            if not self.enable_summary:
                buffer.state.running_summary = ""
            trace.append(
                {
                    "event": "retrieve",
                    "hop": hop,
                    "query": query,
                    "frames": [{"frame_id": f.frame_id, "score": s} for f, s in retrieved],
                }
            )
            pred = self.answerer.answer(example, buffer.frames(), buffer.render_text())
            trace.append(
                {
                    "event": "tentative_answer",
                    "hop": hop,
                    "answer": pred.answer,
                    "answer_index": pred.answer_index,
                    "confidence": pred.confidence,
                }
            )
            if max(probe.sufficiency, pred.confidence) >= self.sufficiency_threshold and len(buffer.frames()) >= 2:
                break

        if pred is None:
            pred = self.answerer.answer(example, buffer.frames(), buffer.render_text())
        return make_result(
            example,
            self.name,
            self.budget,
            pred,
            buffer.frames(),
            buffer.probes(),
            num_retrievals,
            trace,
            metadata={"running_summary": buffer.state.running_summary},
        )

    def _retrieve(
        self,
        example: QAExample,
        query: str,
        k: int,
        exclude: Set[str],
    ) -> List[tuple]:
        """Retrieval hook. Subclasses override to change the relevance signal.

        Base curator uses caption-lexical relevance over the shared frame pool.
        """
        return self.retriever.retrieve(example.video_id, query, k, exclude_ids=exclude)

    def _post_retrieve(self, buffer: RunningContextBuffer, retrieved: List[tuple], hop: int) -> None:
        """Hook after evidence frames are added. Base curator keeps them as pixels (no-op)."""
        return None

    def _pre_loop(self, example: QAExample, buffer: RunningContextBuffer, trace: List[Dict[str, Any]]) -> None:
        """Hook before the probe/retrieve loop, after uniform seeding. Base curator: no-op."""
        return None

    def _probe(self, example: QAExample, buffer: RunningContextBuffer, hop: int) -> ContextProbe:
        held = [
            f"t={frame.timestamp:.1f}s:{frame.caption[:80]}"
            for frame in buffer.frames()
            if frame.caption
        ]
        if not buffer.frames():
            suff = 0.05
        else:
            context = " ".join(frame.caption for frame in buffer.frames())
            suff = min(0.9, 0.18 + 0.14 * len(buffer.frames()) + lexical_score(example.question, context))
        next_action = "answer" if suff >= self.sufficiency_threshold else "retrieve"
        missing = self._missing_moment(example, buffer)
        return ContextProbe(
            held_moments=held,
            missing_moment=missing,
            sufficiency=suff,
            next_action=next_action,
            rationale="scripted probe over current captions and frame budget",
        )

    @staticmethod
    def _missing_moment(example: QAExample, buffer: RunningContextBuffer) -> str:
        context_tokens = set()
        for frame in buffer.frames():
            context_tokens.update(frame_text(frame).lower().split())
        question_terms = [tok for tok in example.question.split() if tok.lower().strip(".,?") not in context_tokens]
        option_terms = " ".join(example.options)
        core = " ".join(question_terms[:12]) if question_terms else example.question
        return f"Find the video moment that resolves: {core}. Candidate details: {option_terms[:160]}"

    @staticmethod
    def _query(example: QAExample, probe: ContextProbe, buffer: RunningContextBuffer) -> str:
        last_context = " ".join(frame.caption for frame in buffer.frames()[-2:])
        return " ".join([example.question, " ".join(example.options), probe.missing_moment, last_context])


class CuratorSummaryOffPolicy(ScriptedCuratorPolicy):
    name = "curator_summary_off"

    def __init__(
        self,
        index: FrameIndex,
        answerer: BaseAnswerer,
        budget: int,
        max_hops: int = 3,
        retrieve_per_hop: int = 4,
        seed_uniform: int = 2,
        sufficiency_threshold: float = 0.68,
    ):
        super().__init__(
            index=index,
            answerer=answerer,
            budget=budget,
            max_hops=max_hops,
            retrieve_per_hop=retrieve_per_hop,
            seed_uniform=seed_uniform,
            sufficiency_threshold=sufficiency_threshold,
            enable_summary=False,
        )


class ForcedCompressCuratorPolicy(ScriptedCuratorPolicy):
    """E-K compression-fidelity: force-COMPRESS every retrieved evidence frame (caption kept in the
    running summary, pixels dropped). Only the coarse seed thumbnails remain as images. Contrasted
    with the KEEP baseline (scripted_curator) it isolates *what survives textualization* — a
    buffer-level operation no frame-selection method can express.
    """

    name = "forced_compress"

    def _post_retrieve(self, buffer: RunningContextBuffer, retrieved: List[tuple], hop: int) -> None:
        for frame, _score in retrieved:
            buffer.compress_frame(frame.frame_id)


class DemoteDontEvictCuratorPolicy(ScriptedCuratorPolicy):
    """N1 'Demote-don't-Evict' (DdE): two-tier memory = K-frame visual working memory plus a
    textual episodic LEDGER that outlives frame eviction, so the frame budget bounds pixels,
    not memory.

    Memory formation: before the probe/retrieve loop, uniformly scan ``scan_n`` frames across
    the whole video and write one dated one-line caption per frame into the ledger via a single
    batched VLM call (chunked at the answerer's max_images); the scanned pixels are NOT kept.
    The scripted loop then fills the K visual slots as usual — a ledger frame can still be
    re-retrieved as pixels (promotion). The answer turn sees K frames + the full ledger.

    Deployable and label-free. Extra cost = ceil(scan_n/max_images) VLM calls, reported in the
    cost table. Contrast with forced_compress (which textualizes the RETRIEVED evidence): DdE
    textualizes the COVERAGE scan and keeps evidence as pixels.
    """

    name = "dde_curator"

    def __init__(
        self,
        index: FrameIndex,
        answerer: BaseAnswerer,
        budget: int,
        max_hops: int = 3,
        retrieve_per_hop: int = 4,
        seed_uniform: int = 2,
        sufficiency_threshold: float = 0.68,
        scan_n: int = 16,
        ledger_chars: int = 6000,
        subtitle_dir: Optional[str] = None,
        subtitle_chars: int = 4000,
        ledger_cache_dir: Optional[str] = None,
        scan_conditioning: bool = False,
    ):
        super().__init__(
            index=index,
            answerer=answerer,
            budget=budget,
            max_hops=max_hops,
            retrieve_per_hop=retrieve_per_hop,
            seed_uniform=seed_uniform,
            sufficiency_threshold=sufficiency_threshold,
            subtitle_dir=subtitle_dir,
            subtitle_chars=subtitle_chars,
        )
        self.scan_n = scan_n
        self.ledger_chars = ledger_chars
        # P5 amortization: scan captions are question-agnostic, so one ledger build per VIDEO
        # can serve every question about it. Cache file = {dir}/{video_id}.scan{scan_n}.json.
        self.ledger_cache_dir = ledger_cache_dir
        # Question-conditioned WRITE: scan captions foreground question-relevant content.
        # A conditioned ledger is question-specific, so it must never touch the video-keyed
        # cache — conditioning trades amortization for accuracy by construction.
        self.scan_conditioning = scan_conditioning

    def _cached_entries(self, video_id: str) -> Optional[List[str]]:
        if not self.ledger_cache_dir:
            return None
        import json as _json
        from pathlib import Path as _Path

        path = _Path(self.ledger_cache_dir) / f"{video_id}.scan{self.scan_n}.json"
        if not path.exists():
            return None
        try:
            data = _json.loads(path.read_text(encoding="utf-8"))
            return list(data) if isinstance(data, list) else None
        except (OSError, ValueError):
            return None

    def _store_entries(self, video_id: str, entries: List[str]) -> None:
        if not self.ledger_cache_dir:
            return
        import json as _json
        from pathlib import Path as _Path

        d = _Path(self.ledger_cache_dir)
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{video_id}.scan{self.scan_n}.json").write_text(
            _json.dumps(entries), encoding="utf-8"
        )

    def _pre_loop(self, example: QAExample, buffer: RunningContextBuffer, trace: List[Dict[str, Any]]) -> None:
        if buffer.state.budget.max_summary_chars < self.ledger_chars:
            buffer.state.budget = BufferBudget(
                max_frames=buffer.state.budget.max_frames,
                max_summary_chars=self.ledger_chars,
            )
        cached = None if self.scan_conditioning else self._cached_entries(example.video_id)
        if cached is not None:
            if cached:
                ledger = "\n".join(cached)
                buffer.state.running_summary = (
                    f"{ledger}\n{buffer.state.running_summary}"
                    if buffer.state.running_summary
                    else ledger
                )
            trace.append({"event": "ledger_scan", "scanned": 0, "captioned": len(cached),
                          "vlm_calls": 0, "cache": "hit"})
            return
        scan = self.sampler.sample(example.video_id, self.scan_n)
        if not scan:
            return
        focus = example.question if self.scan_conditioning else None
        captions = self.answerer.caption_frames(scan, focus=focus)
        entries = []
        for frame, caption in zip(scan, captions):
            if caption:
                entries.append(f"[t={frame.timestamp:.1f}s scanned] {caption}")
        if not self.scan_conditioning:
            self._store_entries(example.video_id, entries)
        if entries:
            ledger = "\n".join(entries)
            buffer.state.running_summary = (
                f"{ledger}\n{buffer.state.running_summary}"
                if buffer.state.running_summary
                else ledger
            )
        # Actual captioning requests: caption_frames chunks at CAPTION_CHUNK (collapse fix),
        # further capped by max_images.
        chunk = min(
            getattr(self.answerer, "CAPTION_CHUNK", getattr(self.answerer, "max_images", 16)),
            getattr(self.answerer, "max_images", 16),
        )
        event = {
            "event": "ledger_scan",
            "scanned": len(scan),
            "captioned": len(entries),
            "vlm_calls": (len(scan) + chunk - 1) // chunk,
        }
        if len(entries) < len(scan):
            # parse shortfall: keep a snippet of the raw captioner output for auditing
            event["raw_snippet"] = str(getattr(self.answerer, "_last_caption_raw", ""))[:400]
        trace.append(event)


class CuratorSufficiencyOffPolicy(ScriptedCuratorPolicy):
    """Ablation: no calibrated stop. Always spends the full hop budget."""

    name = "curator_sufficiency_off"

    def __init__(self, index, answerer, budget, max_hops=3, retrieve_per_hop=4, seed_uniform=2):
        super().__init__(
            index=index,
            answerer=answerer,
            budget=budget,
            max_hops=max_hops,
            retrieve_per_hop=retrieve_per_hop,
            seed_uniform=seed_uniform,
            sufficiency_threshold=2.0,  # unreachable -> never early-stops
        )


class CuratorRetrievalOffPolicy(BasePolicy):
    """Ablation: full memory loop (probe + summary + budget) but NO retrieval.

    Isolates the contribution of evidence retrieval: the buffer is seeded with
    uniformly sampled frames up to budget and the introspective probe runs, but
    the policy can never pull new evidence frames. If this matches uniform, the
    gains come from retrieval; if it beats uniform, the loop/summary help alone.
    """

    name = "curator_retrieval_off"

    def __init__(self, index: FrameIndex, answerer: BaseAnswerer, budget: int, max_hops: int = 3):
        self.index = index
        self.answerer = answerer
        self.budget = budget
        self.max_hops = max_hops
        self.sampler = UniformSampler(index)

    def run(self, example: QAExample) -> PolicyResult:
        buffer = RunningContextBuffer(
            example.question, example.options, BufferBudget(max_frames=self.budget)
        )
        trace: List[Dict[str, Any]] = []
        for frame in self.sampler.sample(example.video_id, self.budget):
            buffer.add_frame(frame, usefulness=0.5, hop=0, note="uniform seed (retrieval-off)")
        probe = ContextProbe(
            held_moments=[f"t={f.timestamp:.1f}s" for f in buffer.frames()],
            missing_moment="",
            sufficiency=0.9,
            next_action="answer",
            rationale="retrieval disabled; answer from seeded frames",
        )
        buffer.add_probe(probe)
        trace.append({"event": "probe", "hop": 1, "probe": probe.to_dict()})
        pred = self.answerer.answer(example, buffer.frames(), buffer.render_text())
        return make_result(
            example, self.name, self.budget, pred, buffer.frames(), buffer.probes(), 0, trace,
            metadata={"running_summary": buffer.state.running_summary},
        )


class AKSPolicy(BasePolicy):
    """AKS-style adaptive keyframe sampling (Tang et al., CVPR'25) on our frame pool.

    Faithful approximation on the shared SigLIP/caption pool: score every frame by
    query relevance, then select via relevance-ranked temporal non-max suppression
    so frames cluster where the question is relevant while preserving coverage.
    Controlled baseline: same frame pool / VLM / budget as the curator.
    """

    name = "aks"

    def __init__(self, index: FrameIndex, answerer: BaseAnswerer, budget: int):
        self.index = index
        self.answerer = answerer
        self.budget = budget

    def run(self, example: QAExample) -> PolicyResult:
        frames = self.index.frames(example.video_id)
        query = " ".join([example.question] + example.options)
        scored = sorted(
            ((f, lexical_score(query, frame_text(f))) for f in frames),
            key=lambda x: x[1], reverse=True,
        )
        if not frames:
            picked: List[FrameRecord] = []
        else:
            span = max(1.0, frames[-1].timestamp - frames[0].timestamp)
            min_gap = span / max(1, self.budget * 2)  # adaptive NMS window
            picked = []
            for frame, _ in scored:
                if len(picked) >= self.budget:
                    break
                if all(abs(frame.timestamp - p.timestamp) >= min_gap for p in picked):
                    picked.append(frame)
            if len(picked) < self.budget:  # backfill coverage if NMS too aggressive
                for frame, _ in scored:
                    if frame not in picked:
                        picked.append(frame)
                    if len(picked) >= self.budget:
                        break
            picked.sort(key=lambda f: f.timestamp)
        buffer = RunningContextBuffer.from_frames(
            example.question, example.options, picked, BufferBudget(max_frames=self.budget)
        )
        pred = self.answerer.answer(example, picked, buffer.render_text())
        return make_result(example, self.name, self.budget, pred, picked, [], 1,
                           [{"event": "aks_select", "kept": len(picked)}])


class AKSOfficialPolicy(AKSPolicy):
    """Exact port of the official AKS selection (Tang et al., CVPR'25;
    github.com/ncTimTang/AKS frame_select.py: mean-std bisection, t1=0.8,
    all_depth=5) on the SAME lexical relevance scores as AKSPolicy. A6
    verification cell: shows the NMS approximation in AKSPolicy is not a
    strawman — only the selection rule differs between the two policies.
    """

    name = "aks_official"

    @staticmethod
    def _select(scores: List[float], budget: int, t1: float = 0.8,
                t2: float = -100.0, all_depth: int = 5) -> List[int]:
        import numpy as np

        n = len(scores)
        if n < budget:
            return list(range(n))
        # Official defaults assume 64-frame budgets; at depth d each kept segment
        # contributes int(budget/2^d) frames, which is 0 when 2^d > budget. Cap the
        # bisection depth so every leaf still contributes >=1 frame.
        all_depth = min(all_depth, max(0, int(np.log2(budget))))
        arr = np.asarray(scores, dtype=float)
        rng = arr.max() - arr.min()
        norm = ((arr - arr.min()) / rng if rng > 0 else np.zeros_like(arr)).tolist()
        pending = [(norm, list(range(n)), 0)]
        kept = []
        while pending:
            score, idxs, depth = pending.pop()
            top = heapq.nlargest(min(budget, len(score)), range(len(score)), score.__getitem__)
            mean_diff = float(np.mean([score[t] for t in top]) - np.mean(score))
            if mean_diff > t1 and float(np.std(score)) > t2:
                kept.append((score, idxs, depth))
            elif depth < all_depth:
                h = len(score) // 2
                pending.append((score[:h], idxs[:h], depth + 1))
                pending.append((score[h:], idxs[h:], depth + 1))
            else:
                kept.append((score, idxs, depth))
        out: List[int] = []
        for score, idxs, depth in kept:
            f_num = int(budget / 2 ** depth)
            top = heapq.nlargest(min(f_num, len(score)), range(len(score)), score.__getitem__)
            out.extend(idxs[t] for t in top)
        return sorted(out)

    def run(self, example: QAExample) -> PolicyResult:
        frames = self.index.frames(example.video_id)
        query = " ".join([example.question] + example.options)
        scores = [lexical_score(query, frame_text(f)) for f in frames]
        picked = [frames[i] for i in self._select(scores, self.budget)][: self.budget]
        buffer = RunningContextBuffer.from_frames(
            example.question, example.options, picked, BufferBudget(max_frames=self.budget)
        )
        pred = self.answerer.answer(example, picked, buffer.render_text())
        return make_result(example, self.name, self.budget, pred, picked, [], 1,
                           [{"event": "aks_official_select", "kept": len(picked)}])


class VideoTreePolicy(BasePolicy):
    """VideoTree-style adaptive breadth-first relevance expansion (Wang et al., CVPR'25).

    Builds a coarse->fine temporal tree: split the video into segments, score each
    segment by its peak query relevance, then expand the highest-relevance segments
    first (one representative frame per expanded segment) until the budget is spent.
    Query-adaptive depth on the shared frame pool; same VLM/budget as the curator.
    """

    name = "videotree"

    def __init__(self, index: FrameIndex, answerer: BaseAnswerer, budget: int, branch: int = 2):
        self.index = index
        self.answerer = answerer
        self.budget = budget
        self.branch = branch

    def run(self, example: QAExample) -> PolicyResult:
        frames = self.index.frames(example.video_id)
        query = " ".join([example.question] + example.options)

        def rep(seg: List[FrameRecord]):
            return max(seg, key=lambda f: lexical_score(query, frame_text(f)))

        def peak(seg: List[FrameRecord]):
            return max((lexical_score(query, frame_text(f)) for f in seg), default=0.0)

        picked: List[FrameRecord] = []
        if frames:
            # frontier of (segment, score); expand most-relevant segment first
            frontier = [frames]
            chosen_ids: Set[str] = set()
            while frontier and len(picked) < self.budget:
                frontier.sort(key=peak, reverse=True)
                seg = frontier.pop(0)
                r = rep(seg)
                if r.frame_id not in chosen_ids:
                    picked.append(r)
                    chosen_ids.add(r.frame_id)
                if len(seg) > 1:  # split into branches for finer resolution
                    size = max(1, len(seg) // self.branch)
                    frontier.extend(seg[i:i + size] for i in range(0, len(seg), size))
            picked.sort(key=lambda f: f.timestamp)
            picked = picked[: self.budget]
        buffer = RunningContextBuffer.from_frames(
            example.question, example.options, picked, BufferBudget(max_frames=self.budget)
        )
        pred = self.answerer.answer(example, picked, buffer.render_text())
        return make_result(example, self.name, self.budget, pred, picked, [], 1,
                           [{"event": "videotree_expand", "kept": len(picked)}])


class FlexibleFramePolicy(BasePolicy):
    """Q-Frame / Flexible-Frame-Selection style budget-adaptive sampling (T2.a baseline).

    Allocates the frame budget in two stages:
    1. Relevance pass: rank all frames by query relevance, pick top-R (= budget // 2).
    2. Diversity pass: from the remaining frames, pick D = budget - R frames via MMR
       over the full temporal range to guarantee coverage.

    This approximates the FlexFrame / Q-Frame idea of separating relevance and coverage
    in a controlled, single-shot (non-iterative) policy on our shared frame pool.
    """

    name = "flexframe"

    def __init__(self, index: FrameIndex, answerer: BaseAnswerer, budget: int):
        self.index = index
        self.answerer = answerer
        self.budget = budget
        self.retriever = MMRRetriever(index, lambda_relevance=0.5)

    def run(self, example: QAExample) -> PolicyResult:
        frames = self.index.frames(example.video_id)
        query = " ".join([example.question] + example.options)

        relevance_k = max(1, self.budget // 2)
        diversity_k = self.budget - relevance_k

        # Stage 1: top-relevance frames
        relevance_ranked = sorted(
            ((f, lexical_score(query, frame_text(f))) for f in frames),
            key=lambda x: x[1],
            reverse=True,
        )
        rel_picked = [f for f, _ in relevance_ranked[:relevance_k]]
        rel_ids = {f.frame_id for f in rel_picked}

        # Stage 2: diverse coverage from remainder
        div_picked: List[FrameRecord] = []
        if diversity_k > 0:
            div_picked = [
                f for f, _ in self.retriever.retrieve(
                    example.video_id, query, diversity_k, exclude_ids=rel_ids
                )
            ]

        picked = rel_picked + div_picked
        picked.sort(key=lambda f: f.timestamp)
        picked = picked[: self.budget]
        buffer = RunningContextBuffer.from_frames(
            example.question, example.options, picked, BufferBudget(max_frames=self.budget)
        )
        pred = self.answerer.answer(example, picked, buffer.render_text())
        return make_result(
            example, self.name, self.budget, pred, picked, [], 1,
            [{"event": "flexframe_select", "relevance_k": len(rel_picked), "diversity_k": len(div_picked)}],
        )


class EmbeddingCuratorPolicy(ScriptedCuratorPolicy):
    """Deployable curator with SigLIP-embedding retrieval relevance (T1.b FEP fix).

    Identical loop to the scripted curator, but retrieval relevance comes from the
    shared SigLIP space — cosine(query_embedding, frame_embedding) — instead of
    caption-lexical overlap, when a precomputed query embedding is available in
    ``example.metadata['query_embedding']`` (produced by the SigLIP-so400m re-index).

    This is **label-free**: it never reads gold evidence spans, so it remains a valid
    controlled-comparison / headline method. The goal is to land retrieval on the true
    evidence moments more often, lifting FEP above uniform without leaking labels.
    When no query embedding is present it degrades gracefully to lexical relevance
    (i.e. behaves like the scripted curator).
    """

    name = "curator_siglip"

    def _retrieve(self, example: QAExample, query: str, k: int, exclude: Set[str]) -> List[tuple]:
        q_emb = query_embedding_for(example)
        return self.retriever.retrieve(
            example.video_id, query, k, exclude_ids=exclude, query_embedding=q_emb
        )


class LearnedCuratorPolicy(EmbeddingCuratorPolicy):
    """GRPO/SFT LoRA adapter drives probe/retrieve/answer instead of heuristics."""

    name = "learned_curator"

    def __init__(
        self,
        index: FrameIndex,
        answerer: BaseAnswerer,
        budget: int,
        action_policy: Any,
        max_hops: int = 4,
        retrieve_per_hop: int = 6,
        seed_uniform: int = 2,
        sufficiency_threshold: float = 0.72,
    ):
        super().__init__(
            index=index,
            answerer=answerer,
            budget=budget,
            max_hops=max_hops,
            retrieve_per_hop=retrieve_per_hop,
            seed_uniform=seed_uniform,
            sufficiency_threshold=sufficiency_threshold,
        )
        self.action_policy = action_policy

    def run(self, example: QAExample) -> PolicyResult:
        buffer = RunningContextBuffer(
            example.question,
            example.options,
            BufferBudget(max_frames=self.budget),
        )
        trace: List[Dict[str, Any]] = []
        for frame in self.sampler.sample(example.video_id, min(self.seed_uniform, self.budget)):
            buffer.add_frame(frame, usefulness=0.25, hop=0, note="coarse thumbnail prior")

        num_retrievals = 0
        pred = None
        hop = 1
        while hop <= self.max_hops:
            held = [
                f"t={frame.timestamp:.1f}s:{(frame.caption or frame.frame_id)[:80]}"
                for frame in buffer.frames()
            ]
            messages = self.action_policy.build_turn_messages(
                example.question,
                list(example.options),
                held,
                buffer.state.running_summary,
                self.budget,
                hop,
            )
            action = self.action_policy.generate(messages)
            if action is None:
                trace.append({"event": "action_parse_fail", "hop": hop})
                hop += 1
                continue

            act = str(action.get("action", "")).lower()
            trace.append({"event": "learned_action", "hop": hop, "action": action})

            if act == "probe":
                probe = ContextProbe(
                    held_moments=action.get("held_moments", held),
                    missing_moment=str(action.get("missing_moment", "")),
                    sufficiency=float(action.get("sufficiency", 0.0)),
                    next_action="answer"
                    if float(action.get("sufficiency", 0.0)) >= self.sufficiency_threshold
                    else "retrieve",
                    rationale="learned probe",
                )
                buffer.add_probe(probe)
                trace.append({"event": "probe", "hop": hop, "probe": probe.to_dict()})
                if probe.next_action == "answer":
                    pred = self.answerer.answer(example, buffer.frames(), buffer.render_text())
                    trace.append(
                        {
                            "event": "tentative_answer",
                            "hop": hop,
                            "answer": pred.answer,
                            "answer_index": pred.answer_index,
                            "confidence": pred.confidence,
                        }
                    )
                    if max(probe.sufficiency, pred.confidence) >= self.sufficiency_threshold:
                        break
                hop += 1
                continue

            if act == "retrieve":
                query = str(action.get("query", "")).strip() or example.question
                exclude: Set[str] = set(buffer.selected_ids())
                retrieved = self._retrieve(example, query, self.retrieve_per_hop, exclude)
                num_retrievals += 1
                for frame, score in retrieved:
                    usefulness = max(score, lexical_score(example.question, frame_text(frame)))
                    buffer.add_frame(frame, usefulness=usefulness, hop=hop, note=f"query={query}")
                trace.append(
                    {
                        "event": "retrieve",
                        "hop": hop,
                        "query": query,
                        "frames": [{"frame_id": f.frame_id, "score": s} for f, s in retrieved],
                    }
                )
                pred = self.answerer.answer(example, buffer.frames(), buffer.render_text())
                trace.append(
                    {
                        "event": "tentative_answer",
                        "hop": hop,
                        "answer": pred.answer,
                        "answer_index": pred.answer_index,
                        "confidence": pred.confidence,
                    }
                )
                if pred.confidence >= self.sufficiency_threshold and len(buffer.frames()) >= 2:
                    break
                hop += 1
                continue

            if act == "answer":
                idx = action.get("answer_index")
                ans = str(action.get("answer", ""))
                conf = float(action.get("confidence", action.get("sufficiency", 0.5)))
                if idx is not None:
                    try:
                        idx = int(idx)
                    except (TypeError, ValueError):
                        idx = None
                if idx is not None and 0 <= idx < len(example.options):
                    ans = example.options[idx]
                elif idx is None and example.options:
                    idx = next((i for i, o in enumerate(example.options) if o == ans), None)
                pred = AnswerPrediction(
                    answer=ans,
                    answer_index=idx,
                    confidence=conf,
                    raw={"learned_action": action},
                )
                return make_result(
                    example,
                    self.name,
                    self.budget,
                    pred,
                    buffer.frames(),
                    buffer.probes(),
                    num_retrievals,
                    trace,
                    metadata={"running_summary": buffer.state.running_summary},
                )

            hop += 1

        if pred is None:
            pred = self.answerer.answer(example, buffer.frames(), buffer.render_text())
        return make_result(
            example,
            self.name,
            self.budget,
            pred,
            buffer.frames(),
            buffer.probes(),
            num_retrievals,
            trace,
            metadata={"running_summary": buffer.state.running_summary},
        )


class SelfConsistencyCuratorPolicy(LearnedCuratorPolicy):
    """N4: test-time curation scaling. Runs k stochastic curation trajectories (sampled
    action turns, temperature>0) and majority-votes the final answer. Agreement fraction
    becomes the confidence, giving a selective-prediction signal for free. Eval-only;
    the answerer stays deterministic — diversity comes from the curator's sampled
    probe/retrieve/answer decisions. Cost = k full curation loops (reported in metadata).
    """

    name = "learned_curator_sc"

    def __init__(
        self,
        index: FrameIndex,
        answerer: BaseAnswerer,
        budget: int,
        action_policy: Any,
        max_hops: int = 4,
        retrieve_per_hop: int = 6,
        seed_uniform: int = 2,
        sufficiency_threshold: float = 0.72,
        num_trajectories: int = 3,
        sc_temperature: float = 0.7,
    ):
        super().__init__(
            index=index,
            answerer=answerer,
            budget=budget,
            action_policy=action_policy,
            max_hops=max_hops,
            retrieve_per_hop=retrieve_per_hop,
            seed_uniform=seed_uniform,
            sufficiency_threshold=sufficiency_threshold,
        )
        self.num_trajectories = num_trajectories
        self.sc_temperature = sc_temperature

    def run(self, example: QAExample) -> PolicyResult:
        prev_temperature = getattr(self.action_policy, "temperature", 0.0)
        self.action_policy.temperature = self.sc_temperature
        run_once = super().run  # bind outside the loop: zero-arg super() breaks in comprehensions
        try:
            trajectories = [run_once(example) for _ in range(self.num_trajectories)]
        finally:
            self.action_policy.temperature = prev_temperature

        votes = [t.answer_index for t in trajectories]
        tally: Dict[Any, int] = {}
        for v in votes:
            tally[v] = tally.get(v, 0) + 1
        # majority answer; ties broken by highest mean confidence among tied indices
        best = max(
            tally,
            key=lambda v: (
                tally[v],
                sum(t.confidence for t in trajectories if t.answer_index == v)
                / max(1, tally[v]),
            ),
        )
        winner = next(t for t in trajectories if t.answer_index == best)
        agreement = tally[best] / len(votes)
        metadata = dict(winner.metadata)
        metadata.update(
            {
                "sc_votes": votes,
                "sc_agreement": agreement,
                "sc_confidences": [t.confidence for t in trajectories],
                "sc_total_frames": sum(t.num_frames for t in trajectories),
                "sc_total_retrievals": sum(t.num_retrievals for t in trajectories),
                "sc_temperature": self.sc_temperature,
            }
        )
        return PolicyResult(
            qid=winner.qid,
            video_id=winner.video_id,
            policy=self.name,
            budget=winner.budget,
            answer=winner.answer,
            answer_index=winner.answer_index,
            confidence=agreement,
            selected_frame_ids=winner.selected_frame_ids,
            selected_timestamps=winner.selected_timestamps,
            probes=winner.probes,
            num_retrievals=winner.num_retrievals,
            num_frames=winner.num_frames,
            trace=winner.trace,
            metadata=metadata,
        )


class OracleProbeCuratorPolicy(ScriptedCuratorPolicy):
    """NON-DEPLOYABLE upper bound: curator whose probe/query reads gold evidence spans.

    This is an analysis ceiling, NOT a controlled-comparison method and NOT a headline
    result. It quantifies how high curator FEP/accuracy could reach if the
    missing-moment probe perfectly named the gold temporal window. Group it with
    ``oracle_span`` — it leaks test labels and must never appear in the headline tables.

    Use it to bound the gap between the deployable ``curator_siglip`` FEP and the best
    achievable FEP given the same loop.
    """

    name = "curator_oracle_probe"

    def _probe(self, example: QAExample, buffer: RunningContextBuffer, hop: int) -> ContextProbe:
        base_probe = super()._probe(example, buffer, hop)
        if not example.evidence_spans:
            return base_probe
        span_labels = " ".join(
            f"[{span.start:.0f}s-{span.end:.0f}s {span.label}]" for span in example.evidence_spans
        )
        missing = (
            f"Find evidence between {span_labels}. "
            f"Question: {example.question[:120]}. "
            f"Candidates: {' '.join(example.options)[:120]}"
        )
        return ContextProbe(
            held_moments=base_probe.held_moments,
            missing_moment=missing,
            sufficiency=base_probe.sufficiency,
            next_action=base_probe.next_action,
            rationale="ORACLE probe (reads gold spans) — analysis ceiling only",
        )

    @staticmethod
    def _query(example: QAExample, probe: ContextProbe, buffer: RunningContextBuffer) -> str:
        last_context = " ".join(frame.caption for frame in buffer.frames()[-2:])
        gold_hint = " ".join(
            f"{span.start:.0f}s {span.end:.0f}s {span.label}"
            for span in example.evidence_spans
        ) if example.evidence_spans else ""
        return " ".join(filter(None, [
            example.question,
            " ".join(example.options),
            gold_hint,
            probe.missing_moment,
            last_context,
        ]))


def make_result(
    example: QAExample,
    policy_name: str,
    budget: int,
    pred: Any,
    frames: Sequence[FrameRecord],
    probes: Sequence[ContextProbe],
    num_retrievals: int,
    trace: List[Dict[str, Any]],
    metadata: Dict[str, Any] | None = None,
) -> PolicyResult:
    return PolicyResult(
        qid=example.qid,
        video_id=example.video_id,
        policy=policy_name,
        budget=budget,
        answer=pred.answer,
        answer_index=pred.answer_index,
        confidence=float(pred.confidence),
        selected_frame_ids=[frame.frame_id for frame in frames],
        selected_timestamps=[frame.timestamp for frame in frames],
        probes=[probe.to_dict() for probe in probes],
        num_retrievals=num_retrievals,
        num_frames=len(frames),
        trace=trace,
        metadata=metadata or {},
    )


def build_policy(
    name: str,
    index: FrameIndex,
    answerer: BaseAnswerer,
    budget: int,
    config: Dict[str, Any],
) -> BasePolicy:
    if name == "uniform":
        return UniformPolicy(
            index,
            answerer,
            budget,
            subtitle_dir=config.get("subtitle_dir"),
            subtitle_chars=int(config.get("subtitle_chars", 4000)),
            subtitle_mode=str(config.get("subtitle_mode", "block")),
        )
    if name == "blind":
        return BlindPolicy(index, answerer, budget)
    if name in {"mmr", "mmr_retrieve"}:
        return MMRPolicy(index, answerer, budget)
    if name in {"single", "best_single_frame"}:
        return BestSingleFramePolicy(index, answerer, budget)
    if name == "oracle_span":
        return OracleSpanPolicy(index, answerer, budget)
    if name == "oracle_span_context":
        return OracleSpanContextPolicy(index, answerer, budget)
    if name == "curator_probe_off":
        return CuratorProbeOffPolicy(
            index,
            answerer,
            budget,
            retrieve_per_hop=int(config.get("retrieve_per_hop", 4)),
        )
    if name == "curator_curation_off":
        return CuratorCurationOffPolicy(
            index,
            answerer,
            budget,
            max_hops=int(config.get("max_hops", 3)),
            retrieve_per_hop=int(config.get("retrieve_per_hop", 4)),
        )
    if name == "curator_summary_off":
        return CuratorSummaryOffPolicy(
            index,
            answerer,
            budget,
            max_hops=int(config.get("max_hops", 3)),
            retrieve_per_hop=int(config.get("retrieve_per_hop", 4)),
            seed_uniform=int(config.get("seed_uniform", 2)),
            sufficiency_threshold=float(config.get("sufficiency_threshold", 0.68)),
        )
    if name == "forced_compress":
        return ForcedCompressCuratorPolicy(
            index,
            answerer,
            budget,
            max_hops=int(config.get("max_hops", 3)),
            retrieve_per_hop=int(config.get("retrieve_per_hop", 4)),
            seed_uniform=int(config.get("seed_uniform", 2)),
            sufficiency_threshold=float(config.get("sufficiency_threshold", 0.68)),
        )
    if name in {"dde", "dde_curator"}:
        return DemoteDontEvictCuratorPolicy(
            index,
            answerer,
            budget,
            max_hops=int(config.get("max_hops", 3)),
            retrieve_per_hop=int(config.get("retrieve_per_hop", 4)),
            seed_uniform=int(config.get("seed_uniform", 2)),
            sufficiency_threshold=float(config.get("sufficiency_threshold", 0.68)),
            scan_n=int(config.get("scan_n", 16)),
            ledger_chars=int(config.get("ledger_chars", 6000)),
            subtitle_dir=config.get("subtitle_dir"),
            subtitle_chars=int(config.get("subtitle_chars", 4000)),
            ledger_cache_dir=config.get("ledger_cache_dir"),
            scan_conditioning=bool(config.get("scan_conditioning", False)),
        )
    if name == "curator_sufficiency_off":
        return CuratorSufficiencyOffPolicy(
            index,
            answerer,
            budget,
            max_hops=int(config.get("max_hops", 3)),
            retrieve_per_hop=int(config.get("retrieve_per_hop", 4)),
            seed_uniform=int(config.get("seed_uniform", 2)),
        )
    if name == "curator_retrieval_off":
        return CuratorRetrievalOffPolicy(
            index,
            answerer,
            budget,
            max_hops=int(config.get("max_hops", 3)),
        )
    if name == "aks":
        return AKSPolicy(index, answerer, budget)
    if name == "aks_official":
        return AKSOfficialPolicy(index, answerer, budget)
    if name == "videotree":
        return VideoTreePolicy(index, answerer, budget, branch=int(config.get("branch", 2)))
    if name in {"curator", "scripted_curator"}:
        return ScriptedCuratorPolicy(
            index,
            answerer,
            budget,
            max_hops=int(config.get("max_hops", 3)),
            retrieve_per_hop=int(config.get("retrieve_per_hop", 4)),
            seed_uniform=int(config.get("seed_uniform", 2)),
            sufficiency_threshold=float(config.get("sufficiency_threshold", 0.68)),
            subtitle_dir=config.get("subtitle_dir"),
            subtitle_chars=int(config.get("subtitle_chars", 4000)),
        )
    if name in {"flexframe", "flex_frame", "flexible_frame"}:
        return FlexibleFramePolicy(index, answerer, budget)
    if name in {"curator_siglip", "embedding_curator"}:
        return EmbeddingCuratorPolicy(
            index,
            answerer,
            budget,
            max_hops=int(config.get("max_hops", 3)),
            retrieve_per_hop=int(config.get("retrieve_per_hop", 4)),
            seed_uniform=int(config.get("seed_uniform", 2)),
            sufficiency_threshold=float(config.get("sufficiency_threshold", 0.68)),
        )
    if name in {"learned_curator", "learned_curator_grpo", "trained_curator_lora_grpo"}:
        action_policy = config.get("_action_policy")
        if action_policy is None:
            raise ValueError(f"policy {name} requires _action_policy in policy config")
        return LearnedCuratorPolicy(
            index,
            answerer,
            budget,
            action_policy=action_policy,
            max_hops=int(config.get("max_hops", 4)),
            retrieve_per_hop=int(config.get("retrieve_per_hop", 6)),
            seed_uniform=int(config.get("seed_uniform", 2)),
            sufficiency_threshold=float(config.get("sufficiency_threshold", 0.72)),
        )
    if name in {"learned_curator_sc", "self_consistency_curator"}:
        action_policy = config.get("_action_policy")
        if action_policy is None:
            raise ValueError(f"policy {name} requires _action_policy in policy config")
        return SelfConsistencyCuratorPolicy(
            index,
            answerer,
            budget,
            action_policy=action_policy,
            max_hops=int(config.get("max_hops", 4)),
            retrieve_per_hop=int(config.get("retrieve_per_hop", 6)),
            seed_uniform=int(config.get("seed_uniform", 2)),
            sufficiency_threshold=float(config.get("sufficiency_threshold", 0.72)),
            num_trajectories=int(config.get("num_trajectories", 3)),
            sc_temperature=float(config.get("sc_temperature", 0.7)),
        )
    # NON-DEPLOYABLE analysis ceiling (reads gold spans). Never a headline method.
    if name in {"curator_oracle_probe", "curator_gold_probe", "gold_aware_curator"}:
        return OracleProbeCuratorPolicy(
            index,
            answerer,
            budget,
            max_hops=int(config.get("max_hops", 3)),
            retrieve_per_hop=int(config.get("retrieve_per_hop", 4)),
            seed_uniform=int(config.get("seed_uniform", 2)),
            sufficiency_threshold=float(config.get("sufficiency_threshold", 0.68)),
        )
    raise ValueError(f"unknown policy: {name}")
