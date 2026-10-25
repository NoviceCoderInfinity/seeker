"""Outcome-aware GRPO reward for VID-CURATOR turn traces (verl custom_reward_function).

Reward regimes (VIDCURATOR_REWARD_MODE), compared in the paper's reward-design study:
  - imitation:    score vs the SCRIPTED TEACHER's reference turn (0.1*format + 0.3*action_match
                  + 0.6*field_match). Capped at the teacher -> flat reward (known negative).
  - outcome:      ANSWER turns scored against GOLD (recovered by qid) so the policy can exceed
                  the fallible teacher: 0.85*correct + 0.15*confidence-calibration.
                  probe/retrieve turns keep imitation shaping (no observable single-turn outcome).
  - outcome_eff:  correctness-GATED efficiency. Correct answers earn 1 - 0.2*(frames/budget);
                  wrong answers earn 0 regardless of cost (a wrong-but-cheap answer must never
                  beat a right-but-expensive one -> no early-answer collapse). Optimizes exactly
                  the CE-AUC objective (accuracy per frame). Needs frames_in_buffer/budget in
                  extra_info (added by prep_grpo_parquet.py); falls back to plain outcome if absent.
  - hybrid:       0.5*field_match + 0.5*outcome on answer turns — dense teacher shaping plus the
                  gold signal, for when pure outcome is too sparse to move early in training.
"""

from __future__ import annotations

import json
import os
import re
from difflib import SequenceMatcher
from typing import Any

# --- gold map (qid -> gold answer_index), loaded once ------------------------------------------
_GOLD_MAP: dict[str, int] | None = None


def _gold_map() -> dict[str, int]:
    global _GOLD_MAP
    if _GOLD_MAP is None:
        path = os.environ.get("VIDCURATOR_GOLD_MAP", "data/rl/qid_gold.json")
        try:
            with open(path, encoding="utf-8") as f:
                _GOLD_MAP = {str(k): int(v) for k, v in json.load(f).items()}
        except (OSError, ValueError):
            _GOLD_MAP = {}
    return _GOLD_MAP


def _extract_json(text: str) -> dict[str, Any] | None:
    text = (text or "").strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        pass
    match = re.search(r"\{[\s\S]*\}", text)
    if not match:
        return None
    try:
        obj = json.loads(match.group(0))
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _norm_text(value: Any) -> str:
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _text_sim(a: Any, b: Any) -> float:
    na, nb = _norm_text(a), _norm_text(b)
    if not na and not nb:
        return 1.0
    if not na or not nb:
        return 0.0
    return SequenceMatcher(None, na, nb).ratio()


def _field_score(pred: dict[str, Any], gt: dict[str, Any], action: str) -> float:
    if action == "answer":
        idx_match = float(pred.get("answer_index") == gt.get("answer_index"))
        ans_sim = _text_sim(pred.get("answer"), gt.get("answer"))
        conf_err = abs(float(pred.get("confidence", 0.0)) - float(gt.get("confidence", 0.0)))
        conf_score = max(0.0, 1.0 - conf_err)
        return 0.6 * idx_match + 0.3 * ans_sim + 0.1 * conf_score
    if action == "retrieve":
        return _text_sim(pred.get("query"), gt.get("query"))
    if action == "probe":
        held_sim = _text_sim(pred.get("held_moments"), gt.get("held_moments"))
        miss_sim = _text_sim(pred.get("missing_moment"), gt.get("missing_moment"))
        suff_err = abs(float(pred.get("sufficiency", 0.0)) - float(gt.get("sufficiency", 0.0)))
        suff_score = max(0.0, 1.0 - suff_err)
        return 0.35 * held_sim + 0.45 * miss_sim + 0.2 * suff_score
    return 0.0


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth: str,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Score one curator action JSON. ANSWER turns -> gold outcome; others -> imitation shaping."""
    del data_source, kwargs
    mode = os.environ.get("VIDCURATOR_REWARD_MODE", "outcome").lower()

    pred = _extract_json(solution_str)
    gt = _extract_json(ground_truth)
    if pred is None or gt is None or "action" not in pred or "action" not in gt:
        return {"score": 0.0, "format_valid": 0.0, "action_match": 0.0,
                "field_match": 0.0, "outcome": 0.0}

    format_valid = 1.0
    gt_action = str(gt.get("action", ""))
    action_match = 1.0 if pred.get("action") == gt_action else 0.0
    field_match = _field_score(pred, gt, gt_action) if action_match else 0.0

    # Legacy imitation regime (paper ablation baseline): match the scripted teacher.
    if mode == "imitation":
        score = 0.1 * format_valid + 0.3 * action_match + 0.6 * field_match
        return {"score": float(score), "format_valid": format_valid,
                "action_match": action_match, "field_match": field_match, "outcome": field_match}

    # Outcome regimes: reward the ANSWER turn against GOLD so the policy can exceed the teacher.
    outcome = 0.0
    if gt_action == "answer" and action_match:
        info = extra_info or {}
        qid = str(info.get("qid", ""))
        gold_idx = _gold_map().get(qid)
        if gold_idx is not None:
            correct = float(pred.get("answer_index") == gold_idx)
            conf = max(0.0, min(1.0, float(pred.get("confidence", 0.0) or 0.0)))
            calib = 1.0 - abs(conf - correct)
            if mode in ("outcome_eff", "outcome_loc"):
                # Correctness-gated frame shaping. The penalty coefficient lambda is the KNOB that
                # places the curator on the coverage (lambda>0: spend few frames) vs localization
                # (lambda<0: reward spending the budget to land evidence) side of the boundary.
                #   outcome_eff default lambda=+0.2 (coverage);  outcome_loc default lambda=-0.2
                #   (localization). Override with VIDCURATOR_FRAME_PENALTY for the reward-lambda sweep.
                default_lam = -0.2 if mode == "outcome_loc" else 0.2
                try:
                    lam = float(os.environ.get("VIDCURATOR_FRAME_PENALTY", default_lam))
                except ValueError:
                    lam = default_lam
                frames = info.get("frames_in_buffer")
                budget = info.get("budget")
                if frames and budget:
                    frac = max(0.0, min(1.0, float(frames) / float(budget)))
                    outcome = correct * (1.0 - lam * frac)
                else:
                    outcome = 0.85 * correct + 0.15 * calib
            elif mode == "hybrid":
                outcome = 0.5 * field_match + 0.5 * (0.85 * correct + 0.15 * calib)
            else:  # plain outcome
                outcome = 0.85 * correct + 0.15 * calib
        else:
            outcome = field_match  # gold unavailable -> fall back to imitation for this turn
        task = outcome
    else:
        # probe/retrieve turns have no observable single-turn outcome: keep imitation shaping.
        task = field_match
        # PROBE-TURN frame shaping (the mechanistically-correct lambda; see
        # reports/lambda_invariance_note.md). A per-prompt cost like frames/budget cancels
        # under group-normalized GRPO advantages on the ANSWER turn, because it is constant
        # within a rollout group. Frames are DECIDED here, on the probe turn, where the
        # sampled next_action varies across rollouts — so the shaping term survives
        # normalization and reaches the gradient. lam_p > 0 rewards spending the budget
        # before answering (localization regime); lam_p < 0 rewards stopping early
        # (coverage regime). Off by default.
        if gt_action == "probe" and action_match:
            try:
                lam_p = float(os.environ.get("VIDCURATOR_PROBE_SHAPE", "0"))
            except ValueError:
                lam_p = 0.0
            if lam_p:
                info = extra_info or {}
                frames = info.get("frames_in_buffer")
                budget = info.get("budget")
                next_action = str(pred.get("next_action", "")).strip().lower()
                # frames == 0 (empty buffer) is a valid, maximally-informative state.
                if frames is not None and budget and next_action in ("answer", "retrieve"):
                    frac = max(0.0, min(1.0, float(frames) / float(budget)))
                    fill = (1.0 - frac) if next_action == "retrieve" else frac
                    task = max(0.0, min(1.2, task + lam_p * fill))

    score = 0.1 * format_valid + 0.2 * action_match + 0.7 * task
    return {"score": float(score), "format_valid": format_valid,
            "action_match": action_match, "field_match": field_match, "outcome": float(outcome)}
