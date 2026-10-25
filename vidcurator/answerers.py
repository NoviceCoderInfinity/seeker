"""Answerer interfaces.

The default lexical answerer exists for smoke tests and ablations. Real runs
should point the config at a vLLM/SGLang/OpenAI-compatible multimodal endpoint.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from .retrieval import lexical_score, tokenize
from .types import AnswerPrediction, FrameRecord, QAExample


class BaseAnswerer:
    def answer(
        self,
        example: QAExample,
        frames: Sequence[FrameRecord],
        rendered_context: str,
        subtitle_events: Optional[Sequence[tuple]] = None,
    ) -> AnswerPrediction:
        raise NotImplementedError

    def caption_frames(self, frames: Sequence[FrameRecord], focus: Optional[str] = None) -> List[str]:
        """One short caption per frame, same order. Base fallback: whatever the index holds
        (placeholder captions for decoded pools), so text-only smoke tests still run.
        ``focus``: optional question text — a conditioned WRITE (captions foreground
        question-relevant content); ignored by the lexical fallback."""
        return [frame.caption or "" for frame in frames]


class LexicalAnswerer(BaseAnswerer):
    def answer(
        self,
        example: QAExample,
        frames: Sequence[FrameRecord],
        rendered_context: str,
        subtitle_events: Optional[Sequence[tuple]] = None,
    ) -> AnswerPrediction:
        evidence_context = " ".join(frame.caption for frame in frames if frame.caption).strip()
        context = evidence_context or rendered_context
        if example.options:
            context_tokens = set(tokenize(context))
            question_tokens = set(tokenize(example.question))
            option_tokens = [set(tokenize(option)) for option in example.options]
            common_option_tokens = set.intersection(*option_tokens) if option_tokens else set()
            scores = []
            for idx, option in enumerate(example.options):
                discriminative = option_tokens[idx] - question_tokens - common_option_tokens
                if not discriminative:
                    discriminative = option_tokens[idx] - question_tokens
                hits = len(discriminative & context_tokens)
                miss = len(discriminative - context_tokens)
                discrim_score = hits / max(1.0, len(discriminative) ** 0.5)
                phrase_score = lexical_score(option, context)
                score = discrim_score + 0.25 * phrase_score - 0.08 * miss
                scores.append((idx, score))
            ordered = sorted(scores, key=lambda item: item[1], reverse=True)
            best_idx, best_score = ordered[0]
            margin = best_score - ordered[1][1] if len(ordered) > 1 else best_score
            confidence = min(0.99, max(0.05, 0.35 + 0.35 * best_score + 0.15 * margin))
            return AnswerPrediction(
                answer=example.options[best_idx],
                answer_index=best_idx,
                confidence=confidence,
                raw={"scores": scores, "answerer": "lexical"},
            )

        answer = self._extractive_answer(example, context)
        return AnswerPrediction(answer=answer, answer_index=None, confidence=0.35, raw={"answerer": "lexical"})

    @staticmethod
    def _extractive_answer(example: QAExample, context: str) -> str:
        question_terms = set(tokenize(example.question))
        candidates: List[str] = []
        for phrase in context.replace(".", "\n").replace(";", "\n").splitlines():
            stripped = phrase.strip()
            if not stripped:
                continue
            if set(tokenize(stripped)) - question_terms:
                candidates.append(stripped)
        return candidates[0] if candidates else ""


class OpenAICompatAnswerer(BaseAnswerer):
    """Minimal stdlib client for vLLM/SGLang/OpenAI-compatible chat endpoints."""

    def __init__(
        self,
        endpoint: str,
        model: str,
        api_key_env: str = "OPENAI_API_KEY",
        timeout_seconds: float = 120.0,
        send_images: bool = False,
        max_images: int = 64,
        prompt_style: str = "v2",
        temperature: float = 0.0,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.model = model
        self.api_key_env = api_key_env
        self.timeout_seconds = timeout_seconds
        self.send_images = send_images
        self.max_images = max_images
        # Answer-turn sampling temperature (captioning stays greedy). 0.0 = greedy, the
        # protocol default for every reported cell; >0 only for diagnostic probes.
        self.temperature = temperature
        # "v2": legacy layout (instruction first). "v3": context/images first, question +
        # answer instruction LAST — long ledgers otherwise push the model into continuing
        # the curator protocol instead of answering (lvb-heldout collapse, 2026-07-11).
        self.prompt_style = prompt_style

    def answer(
        self,
        example: QAExample,
        frames: Sequence[FrameRecord],
        rendered_context: str,
        subtitle_events: Optional[Sequence[tuple]] = None,
    ) -> AnswerPrediction:
        if self.prompt_style == "v3" and subtitle_events and self.send_images:
            # Interleaved multi-stream: subtitle lines placed BETWEEN the images at their
            # timestamps, so subtitle<->frame co-occurrence is representable (lvb pilot P2:
            # concatenated streams fuse at chance; the failure is fusion, not availability).
            content = [{"type": "text", "text": f"Context for a video question. {rendered_context}\n"
                        "Chronological evidence: frames interleaved with the subtitles spoken around them."}]
            content += self._interleaved_content(frames[: self.max_images], subtitle_events)
            content.append({"type": "text", "text": self._question_block(example)})
        elif self.prompt_style == "v3":
            context_text = self._context_block(frames, rendered_context)
            question_text = self._question_block(example)
            if self.send_images:
                content = self._multimodal_content(context_text, frames[: self.max_images])
                content.append({"type": "text", "text": question_text})
            else:
                content = f"{context_text}\n{question_text}"
        else:
            prompt = self._prompt(example, frames, rendered_context)
            content = prompt
            if self.send_images:
                content = self._multimodal_content(prompt, frames[: self.max_images])
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": content}],
            "temperature": self.temperature,
            "max_tokens": 128,
        }
        raw = self._post(payload)
        text = raw["choices"][0]["message"]["content"].strip()
        idx = self._parse_option_index(text, example.options)
        answer = example.options[idx] if idx is not None and example.options else text
        return AnswerPrediction(answer=answer, answer_index=idx, confidence=0.5, raw={"response": raw})

    # Qwen2.5-VL collapses to a single video-narration line beyond ~4 images when asked for
    # per-frame captions (verified empirically: 4 -> 4 lines, 8/12/16 -> 1 line), so caption
    # calls are chunked at 4 regardless of max_images.
    CAPTION_CHUNK = 4

    def caption_frames(self, frames: Sequence[FrameRecord], focus: Optional[str] = None) -> List[str]:
        """Batched memory-formation captioning: one VLM call per CAPTION_CHUNK frames, one
        numbered line each. Powers the DdE ledger (budget bounds pixels, not memory).
        ``focus``: optional question text — conditioned WRITE: captions stay factual but
        foreground content relevant to the question (breaks video-keyed cacheability)."""
        captions: List[str] = []
        step = min(self.CAPTION_CHUNK, self.max_images)
        for start in range(0, len(frames), step):
            chunk = list(frames[start : start + step])
            n = len(chunk)
            prompt_lines = [
                f"You see {n} video frames in chronological order.",
                f"Write EXACTLY {n} numbered lines. Line i = one factual caption (8-18 words)",
                "of frame i: visible people, objects, actions, text, scene. No preamble.",
            ]
            if focus:
                prompt_lines.append(
                    f"If a frame shows anything relevant to this question, foreground that "
                    f"in its line (stay factual, do not answer): {focus[:200]}"
                )
            prompt_lines += [
                "Format:",
                "1. <caption of frame 1>",
            ]
            if n > 1:
                prompt_lines.append(f"... up to\n{n}. <caption of frame {n}>")
            content = self._multimodal_content("\n".join(prompt_lines), chunk)
            payload = {
                "model": self.model,
                "messages": [{"role": "user", "content": content}],
                "temperature": 0.0,
                "max_tokens": 40 * len(chunk) + 64,
            }
            try:
                raw = self._post(payload)
                text = raw["choices"][0]["message"]["content"]
            except (RuntimeError, KeyError, IndexError, TypeError):
                # endpoint/format failure -> EMPTY captions (never placeholder junk in the ledger)
                self._last_caption_raw = "<caption request failed>"
                captions.extend("" for _ in chunk)
                continue
            self._last_caption_raw = text
            by_num: Dict[int, str] = {}
            ordered: List[str] = []
            for line in text.splitlines():
                line = line.strip()
                if not line:
                    continue
                numbered = re.match(r"^(\d+)[.)]\s*(.+)$", line)
                if numbered and numbered.group(2).strip():
                    by_num[int(numbered.group(1))] = numbered.group(2).strip()
                    ordered.append(numbered.group(2).strip())
                    continue
                bullet = re.match(r"^[-*]\s+(.+)$", line)
                if bullet:
                    ordered.append(bullet.group(1).strip())
            for pos, frame in enumerate(chunk):
                cap = by_num.get(pos + 1)
                if cap is None and len(ordered) == len(chunk):
                    cap = ordered[pos]  # unnumbered but line count right -> align by order
                captions.append((cap or "").strip())
        return captions

    def _post(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        body = json.dumps(payload).encode("utf-8")
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env)
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        url = f"{self.endpoint}/v1/chat/completions"
        request = urllib.request.Request(url, data=body, headers=headers, method="POST")
        last_exc: Exception | None = None
        for attempt in range(3):
            try:
                with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                    return json.loads(response.read().decode("utf-8"))
            except (urllib.error.URLError, TimeoutError) as exc:
                last_exc = exc
                if attempt < 2:
                    time.sleep(5 * (attempt + 1))
                    continue
        raise RuntimeError(f"model endpoint failed after retries: {url}") from last_exc

    @staticmethod
    def _context_block(frames: Sequence[FrameRecord], rendered_context: str) -> str:
        lines = [
            "Context for a video question (memory ledger, transcript, probe log; the question follows after the frames):",
            rendered_context,
        ]
        if frames:
            lines.append("Chronological frame evidence:")
            for frame in sorted(frames, key=lambda item: item.timestamp):
                lines.append(f"- t={frame.timestamp:.1f}s id={frame.frame_id}: {frame.caption or frame.path}")
        return "\n".join(lines)

    @staticmethod
    def _question_block(example: QAExample) -> str:
        lines = [f"Question: {example.question}"]
        if example.options:
            lines.append("Options:")
            for idx, option in enumerate(example.options):
                letter = chr(ord("A") + idx) if idx < 26 else str(idx)
                lines.append(f"{letter}. {option}")
        lines.append("Answer the multiple-choice question using the frames and context above.")
        lines.append("Choose exactly one option. Return JSON: {\"answer\": \"<letter>\"}.")
        return "\n".join(lines)

    @staticmethod
    def _prompt(example: QAExample, frames: Sequence[FrameRecord], rendered_context: str) -> str:
        lines = [
            "Answer the multiple-choice video question using the supplied frame evidence.",
            "Choose exactly one option. Respond with the option letter (A, B, C, ...).",
            "Return JSON: {\"answer\": \"<letter>\"}.",
            rendered_context,
        ]
        if frames:
            lines.append("Chronological frame evidence:")
            for frame in sorted(frames, key=lambda item: item.timestamp):
                lines.append(f"- t={frame.timestamp:.1f}s id={frame.frame_id}: {frame.caption or frame.path}")
        return "\n".join(lines)

    @staticmethod
    def _multimodal_content(prompt: str, frames: Sequence[FrameRecord]) -> List[Dict[str, Any]]:
        content: List[Dict[str, Any]] = [{"type": "text", "text": prompt}]
        for frame in sorted(frames, key=lambda item: item.timestamp):
            part = OpenAICompatAnswerer._image_part(frame)
            if part is not None:
                content.append(part)
        return content

    @staticmethod
    def _image_part(frame: FrameRecord) -> Optional[Dict[str, Any]]:
        if not frame.path:
            return None
        path = Path(frame.path)
        if not path.exists():
            return None
        mime = mimetypes.guess_type(path.name)[0] or "image/jpeg"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return {
            "type": "image_url",
            "image_url": {"url": f"data:{mime};base64,{encoded}", "detail": "low"},
        }

    @staticmethod
    def _interleaved_content(
        frames: Sequence[FrameRecord], subtitle_events: Sequence[tuple]
    ) -> List[Dict[str, Any]]:
        """Chronological merge: subtitle lines emitted as text parts BETWEEN the frame images
        at their timestamps. Trailing subtitles (after the last frame) are appended."""
        parts: List[Dict[str, Any]] = []
        subs = sorted((float(t), str(txt)) for t, txt in subtitle_events)
        si = 0
        pending: List[str] = []
        for frame in sorted(frames, key=lambda item: item.timestamp):
            while si < len(subs) and subs[si][0] <= frame.timestamp:
                pending.append(f"[t={int(subs[si][0])}s sub] {subs[si][1]}")
                si += 1
            image = OpenAICompatAnswerer._image_part(frame)
            if image is None:
                continue  # keep pending subs for the next renderable frame
            header = f"[frame at t={frame.timestamp:.0f}s]"
            parts.append({"type": "text", "text": "\n".join(pending + [header])})
            parts.append(image)
            pending = []
        tail = pending + [f"[t={int(t)}s sub] {txt}" for t, txt in subs[si:]]
        if tail:
            parts.append({"type": "text", "text": "\n".join(tail)})
        return parts

    @staticmethod
    def _parse_option_index(text: str, options: Sequence[str]) -> Optional[int]:
        candidates = [text.strip()]
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if match:
            candidates.insert(0, match.group(0))
        # Letter range must span the number of options (MLVU has 6 → A-F, not A-E).
        n = max(1, len(options))
        last_letter = chr(ord("A") + min(n, 26) - 1)
        letter_re = re.compile(rf"\b([A-{last_letter}])\b", flags=re.IGNORECASE)
        for candidate in candidates:
            cleaned = candidate.strip().strip("`")
            if cleaned.lower().startswith("json"):
                cleaned = cleaned[4:].strip()
            try:
                parsed = json.loads(cleaned)
                value = parsed.get("answer_index")
                if value is None:
                    value = parsed.get("answer")
                idx = OpenAICompatAnswerer._coerce_option_index(value, options)
                if idx is not None:
                    return idx
            except (json.JSONDecodeError, TypeError, ValueError, AttributeError):
                pass
        letter_match = letter_re.search(text)
        if letter_match:
            idx = ord(letter_match.group(1).upper()) - ord("A")
            if 0 <= idx < len(options):
                return idx
        try:
            idx = int(text.strip())
            if 0 <= idx < len(options):
                return idx
        except (TypeError, ValueError):
            pass
        lowered = text.lower()
        for idx, option in enumerate(options):
            if option.lower() in lowered:
                return idx
        return None

    @staticmethod
    def _coerce_option_index(value: Any, options: Sequence[str]) -> Optional[int]:
        if value is None:
            return None
        if isinstance(value, str):
            stripped = value.strip()
            if len(stripped) == 1 and stripped.isalpha():
                idx = ord(stripped.upper()) - ord("A")
                if 0 <= idx < len(options):
                    return idx
            if stripped.isdigit():
                idx = int(stripped)
                return idx if 0 <= idx < len(options) else None
            lowered = stripped.lower()
            for idx, option in enumerate(options):
                if lowered == option.lower() or lowered in option.lower() or option.lower() in lowered:
                    return idx
            return None
        idx = int(value)
        return idx if 0 <= idx < len(options) else None


def build_answerer(config: Dict[str, Any]) -> BaseAnswerer:
    kind = config.get("kind", "lexical")
    if kind == "lexical":
        return LexicalAnswerer()
    if kind == "openai_compat":
        return OpenAICompatAnswerer(
            endpoint=str(config["endpoint"]),
            model=str(config["model"]),
            api_key_env=str(config.get("api_key_env", "OPENAI_API_KEY")),
            timeout_seconds=float(config.get("timeout_seconds", 120.0)),
            send_images=bool(config.get("send_images", False)),
            max_images=int(config.get("max_images", 64)),
            prompt_style=str(config.get("prompt_style", "v2")),
            temperature=float(config.get("temperature", 0.0)),
        )
    raise ValueError(f"unknown answerer kind: {kind}")
