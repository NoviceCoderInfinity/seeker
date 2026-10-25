"""Reward helpers and GRPO training scaffolding for Stage-1 LoRA-GRPO.

Architecture (from §11 of PROJECT_MASTER_PLAN.md):
  SFT cold-start from scripted traces → GRPO fine-tuning
  Reward = correctness − frame/retrieval/probe cost + evidence-precision + calibrated-stop

This module provides:
  1. RewardWeights / compute_reward  — per-rollout scalar reward
  2. extract_sft_traces              — convert scripted curator PolicyResults → SFT JSONL
  3. GRPOTrainerConfig               — config dataclass for the training loop
  4. build_grpo_reward_fn            — returns a callable for verl/trl GRPO trainers
  5. compute_group_rewards           — group-normalised advantage for GRPO (mean-zero, std-1)

The actual LoRA-GRPO trainer loop is invoked via verl (see configs/grpo_stage1.json).
This module is the reward bridge between the vidcurator environment and the trainer.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from .metrics import frame_evidence_pr, is_correct
from .types import PolicyResult, QAExample


# ---------------------------------------------------------------------------
# Reward definition
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class RewardWeights:
    correct: float = 1.0
    num_frames: float = 0.1
    num_retrievals: float = 0.05
    num_probes: float = 0.05
    frame_evidence_precision: float = 0.3
    calibrated_stop: float = 0.2
    format_valid: float = 0.1

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "RewardWeights":
        fields = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        return cls(**{k: float(v) for k, v in d.items() if k in fields})


def compute_reward(
    result: PolicyResult,
    example: QAExample,
    weights: RewardWeights | None = None,
    max_frames: int = 32,
    max_retrievals: int = 6,
    max_probes: int = 4,
) -> Dict[str, Any]:
    """Compute a scalar GRPO reward for one (rollout, example) pair.

    Reward components:
      +correct         — 1.0 if answer is correct
      −num_frames      — penalise excess frames (normalised to [0,1])
      −num_retrievals  — penalise retrieval calls (normalised)
      −num_probes      — penalise probe calls (normalised)
      +fep             — frame-evidence precision (vs gold spans; 0 if no spans)
      +calibration     — 1 − |confidence − correctness|
      +format_valid    — 1 if answer or answer_index is non-empty
    """
    weights = weights or RewardWeights()
    correct = 1.0 if is_correct(result, example) else 0.0
    fep, _ = frame_evidence_pr(result.selected_timestamps, example)
    fep_value = fep if fep is not None else 0.0
    calibration = 1.0 - abs(result.confidence - correct)
    format_valid = 1.0 if (result.answer or result.answer_index is not None) else 0.0

    norm_frames = min(1.0, result.num_frames / max(1, max_frames))
    norm_retrievals = min(1.0, result.num_retrievals / max(1, max_retrievals))
    norm_probes = min(1.0, len(result.probes) / max(1, max_probes))

    reward = (
        weights.correct * correct
        - weights.num_frames * norm_frames
        - weights.num_retrievals * norm_retrievals
        - weights.num_probes * norm_probes
        + weights.frame_evidence_precision * fep_value
        + weights.calibrated_stop * calibration
        + weights.format_valid * format_valid
    )
    return {
        "reward": reward,
        "correct": correct,
        "num_frames": result.num_frames,
        "num_retrievals": result.num_retrievals,
        "num_probes": len(result.probes),
        "frame_evidence_precision": fep,
        "calibrated_stop": calibration,
        "format_valid": format_valid,
    }


# ---------------------------------------------------------------------------
# Group-normalised advantage (GRPO)
# ---------------------------------------------------------------------------

def compute_group_rewards(
    rewards: Sequence[float],
    eps: float = 1e-8,
) -> List[float]:
    """Normalise a group of rollout rewards to zero-mean, unit-variance advantages.

    This is the standard GRPO advantage normalisation used by verl/trl.
    If all rewards are identical (zero variance), returns zeros.
    """
    n = len(rewards)
    if n == 0:
        return []
    mean = sum(rewards) / n
    variance = sum((r - mean) ** 2 for r in rewards) / n
    std = math.sqrt(variance + eps)
    return [(r - mean) / std for r in rewards]


# ---------------------------------------------------------------------------
# SFT trace extraction (for cold-start)
# ---------------------------------------------------------------------------

# Schema for the SFT tool-call format fed to the VLM:
#   system: role description + tool schema
#   user:   {question, options, held_frames, running_summary}
#   assistant: {action: "probe"|"retrieve"|"answer", ...fields}
#
# Each scripted curator PolicyResult is unrolled into a sequence of turns.

_SYSTEM_PROMPT = """\
You are a video-question-answering curator agent.
You operate a bounded frame buffer (budget K frames) and must answer a multiple-choice
question about a long video by selectively retrieving evidence.

At each turn you emit exactly ONE of the following JSON actions:

{"action": "probe", "held_moments": [...], "missing_moment": "...", "sufficiency": 0.0..1.0}
  — Introspect your current evidence. Describe what you have and what you still need.

{"action": "retrieve", "query": "..."}
  — Request retrieval of the next most-useful evidence moment.

{"action": "answer", "answer_index": <int>, "answer": "...", "confidence": 0.0..1.0}
  — Commit your final answer when evidence is sufficient.

Rules:
- Never exceed the frame budget K.
- Emit answer only when sufficiency ≥ 0.68 or you have reached the turn limit.
- Output ONLY the JSON object, no prose.
"""


def _make_user_turn(
    question: str,
    options: List[str],
    held_frames: List[str],
    running_summary: str,
    budget: int,
    hop: int,
) -> str:
    opts_text = "\n".join(f"{i}. {o}" for i, o in enumerate(options))
    held_text = "\n".join(f"  - {m}" for m in held_frames) if held_frames else "  (none)"
    summary_block = f"\nRunning summary:\n{running_summary}" if running_summary else ""
    return (
        f"Turn {hop} | Budget K={budget}\n"
        f"Question: {question}\nOptions:\n{opts_text}\n"
        f"Held frames:\n{held_text}{summary_block}"
    )


def extract_sft_traces(
    result: PolicyResult,
    example: QAExample,
    only_correct: bool = True,
) -> List[Dict[str, Any]]:
    """Convert one scripted curator PolicyResult into a sequence of SFT conversation turns.

    Returns a list of conversation dicts (one per hop), each with:
      {"messages": [{"role": "system", ...}, {"role": "user", ...}, {"role": "assistant", ...}]}

    Only emits traces for correct answers when only_correct=True (avoids poisoning
    the SFT cold-start with wrong reasoning chains).
    """
    if only_correct and not is_correct(result, example):
        return []

    traces: List[Dict[str, Any]] = []
    probe_events = [e for e in result.trace if e.get("event") == "probe"]
    retrieve_events = {e["hop"]: e for e in result.trace if e.get("event") == "retrieve"}
    tentative_events = {e["hop"]: e for e in result.trace if e.get("event") == "tentative_answer"}
    running_summary = result.metadata.get("running_summary", "")

    held_frames: List[str] = []
    for event in probe_events:
        hop = event["hop"]
        probe = event["probe"]
        user_text = _make_user_turn(
            example.question,
            example.options,
            held_frames,
            running_summary,
            result.budget,
            hop,
        )
        # Probe turn
        assistant_probe = json.dumps({
            "action": "probe",
            "held_moments": probe.get("held_moments", []),
            "missing_moment": probe.get("missing_moment", ""),
            "sufficiency": probe.get("sufficiency", 0.0),
        }, ensure_ascii=False)
        traces.append({
            "messages": [
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": assistant_probe},
            ],
            "metadata": {"qid": result.qid, "hop": hop, "action": "probe", "policy": result.policy},
        })

        # Retrieve turn (if the probe triggered retrieval at this hop)
        if hop in retrieve_events:
            ret = retrieve_events[hop]
            frames_retrieved = ret.get("frames", [])
            held_frames = [f"id={f['frame_id']}" for f in frames_retrieved]
            retrieve_user = _make_user_turn(
                example.question,
                example.options,
                held_frames,
                running_summary,
                result.budget,
                hop,
            )
            assistant_retrieve = json.dumps({
                "action": "retrieve",
                "query": ret.get("query", ""),
            }, ensure_ascii=False)
            traces.append({
                "messages": [
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": retrieve_user},
                    {"role": "assistant", "content": assistant_retrieve},
                ],
                "metadata": {"qid": result.qid, "hop": hop, "action": "retrieve", "policy": result.policy},
            })

        # Check if we should emit the final answer at this hop
        if probe.get("next_action") == "answer" or hop not in retrieve_events:
            break

    # Final answer turn
    answer_user = _make_user_turn(
        example.question,
        example.options,
        held_frames,
        running_summary,
        result.budget,
        len(probe_events) + 1,
    )
    assistant_answer = json.dumps({
        "action": "answer",
        "answer_index": result.answer_index,
        "answer": result.answer,
        "confidence": round(result.confidence, 3),
    }, ensure_ascii=False)
    traces.append({
        "messages": [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": answer_user},
            {"role": "assistant", "content": assistant_answer},
        ],
        "metadata": {"qid": result.qid, "hop": len(probe_events) + 1, "action": "answer", "policy": result.policy},
    })
    return traces


def export_sft_dataset(
    results: Sequence[PolicyResult],
    examples: Sequence[QAExample],
    output_path: str | Path,
    only_correct: bool = True,
    policy_filter: str = "scripted_curator",
) -> int:
    """Export SFT training data from a set of PolicyResults.

    Filters to a single policy (default: scripted_curator), extracts turn-level
    traces, and writes them as JSONL to output_path.  Returns the number of rows.
    """
    by_qid = {ex.qid: ex for ex in examples}
    rows_written = 0
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as fh:
        for result in results:
            if policy_filter and result.policy != policy_filter:
                continue
            example = by_qid.get(result.qid)
            if example is None:
                continue
            for trace in extract_sft_traces(result, example, only_correct=only_correct):
                fh.write(json.dumps(trace, ensure_ascii=False) + "\n")
                rows_written += 1
    return rows_written


# ---------------------------------------------------------------------------
# GRPO reward function bridge (for verl / trl integration)
# ---------------------------------------------------------------------------

@dataclass
class GRPOTrainerConfig:
    """Config for the LoRA-GRPO training loop (see configs/grpo_stage1.json)."""

    base_model: str = "Qwen/Qwen2.5-VL-7B-Instruct"
    train_data: str = ""
    eval_data: str = ""
    output_root: str = "runs/grpo"
    lora_rank: int = 32
    lora_alpha: int = 64
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj"])
    global_batch_size: int = 64
    rollout_n: int = 4          # rollouts per prompt
    max_turns: int = 4
    budget_curriculum: List[int] = field(default_factory=lambda: [4, 8, 16])
    reward_weights: RewardWeights = field(default_factory=RewardWeights)
    max_frames: int = 32

    @classmethod
    def from_json(cls, path: str | Path) -> "GRPOTrainerConfig":
        raw = json.loads(Path(path).read_text())
        weights = RewardWeights.from_dict(raw.get("reward_weights", {}))
        verl = raw.get("verl", {})
        return cls(
            base_model=raw.get("base_models", [cls.base_model])[0],
            train_data=raw.get("train_data", ""),
            eval_data=raw.get("eval_data", ""),
            output_root=raw.get("output_root", cls.output_root),
            lora_rank=int(verl.get("lora_rank", cls.lora_rank)),
            global_batch_size=int(verl.get("global_batch_size", cls.global_batch_size)),
            rollout_n=int(verl.get("rollout_n", cls.rollout_n)),
            max_turns=int(verl.get("max_turns", cls.max_turns)),
            budget_curriculum=raw.get("budget_curriculum", cls.budget_curriculum),
            reward_weights=weights,
        )


def build_grpo_reward_fn(
    examples: Sequence[QAExample],
    weights: RewardWeights | None = None,
    max_frames: int = 32,
) -> Callable[[PolicyResult], Dict[str, Any]]:
    """Return a callable that maps (PolicyResult) → reward dict.

    This is the bridge between the vidcurator evaluation environment and
    the GRPO trainer (verl/trl).  The trainer calls this function on each
    rollout to obtain the scalar reward signal.

    Usage (inside verl reward worker):
        reward_fn = build_grpo_reward_fn(eval_examples, weights=cfg.reward_weights)
        reward = reward_fn(rollout_result)["reward"]
    """
    by_qid = {ex.qid: ex for ex in examples}
    _weights = weights or RewardWeights()

    def reward_fn(result: PolicyResult) -> Dict[str, Any]:
        example = by_qid.get(result.qid)
        if example is None:
            return {"reward": 0.0, "error": f"qid {result.qid} not found"}
        return compute_reward(result, example, weights=_weights, max_frames=max_frames)

    return reward_fn


def compute_batch_rewards(
    results: Sequence[PolicyResult],
    examples: Sequence[QAExample],
    weights: RewardWeights | None = None,
    max_frames: int = 32,
    group_normalise: bool = True,
) -> List[Dict[str, Any]]:
    """Compute rewards for a batch of rollouts, with optional group normalisation.

    When group_normalise=True, rewards are mean-zero unit-variance within the batch
    (GRPO advantage), stored as result["advantage"].
    """
    by_qid = {ex.qid: ex for ex in examples}
    _weights = weights or RewardWeights()
    reward_dicts = []
    for result in results:
        example = by_qid.get(result.qid)
        if example is None:
            reward_dicts.append({"reward": 0.0, "advantage": 0.0})
        else:
            reward_dicts.append(compute_reward(result, example, weights=_weights, max_frames=max_frames))

    if group_normalise and reward_dicts:
        raw_rewards = [d["reward"] for d in reward_dicts]
        advantages = compute_group_rewards(raw_rewards)
        for d, adv in zip(reward_dicts, advantages):
            d["advantage"] = adv

    return reward_dicts
