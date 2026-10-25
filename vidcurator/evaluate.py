"""Experiment runner."""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Dict, List

from .answerers import build_answerer
from .io import ensure_dir, load_examples, load_frames, read_jsonl, save_results, write_jsonl, write_json
from .metrics import summarize_results
from .policies import build_policy
from .retrieval import FrameIndex, retrieval_recall_at_k
from .types import PolicyResult


def _shard_path(output_dir: Path, budget: int, policy_name: str) -> Path:
    return output_dir / "_shards" / f"b{budget}_{policy_name}.jsonl"


def expected_shard_count(budgets: List[int], policy_names: List[str]) -> int:
    n = 0
    for budget in budgets:
        for policy_name in policy_names:
            if policy_name == "best_single_frame" and budget != budgets[0]:
                continue
            n += 1
    return n


def _load_completed_shard(path: Path, expected: int) -> List[PolicyResult] | None:
    """Return cached results if the shard exists and looks complete, else None."""
    if not path.exists():
        return None
    try:
        rows = read_jsonl(path)
    except Exception:
        return None
    if len(rows) != expected:
        return None
    try:
        return [PolicyResult.from_dict(row) for row in rows]
    except Exception:
        return None


def _preflight_frame_index(config: Dict[str, Any], index: "FrameIndex", examples: list) -> None:
    """Fail loudly if the frame index is empty or does not cover the QA videos.

    The 0-byte MLVU index shipped silently and every policy answered from text
    alone (avg_frames=0). Refuse to start rather than burn GPU on a dead index.
    Set VIDCURATOR_SKIP_INDEX_CHECK=1 for intentionally text-only runs.
    """
    if os.environ.get("VIDCURATOR_SKIP_INDEX_CHECK", "").lower() in ("1", "true", "yes"):
        return
    frames_path = config.get("frames_path", "<none>")
    if not examples:
        return
    probe = examples[: min(25, len(examples))]
    covered = sum(1 for ex in probe if index.frames(ex.video_id))
    if covered == 0:
        raise RuntimeError(
            f"frame index '{frames_path}' returns 0 frames for all {len(probe)} probed "
            f"videos (e.g. video_id={probe[0].video_id!r}). Index is empty or the "
            f"video_id keys do not match the QA split. Fix the index before evaluating "
            f"(or set VIDCURATOR_SKIP_INDEX_CHECK=1 for a deliberate text-only run)."
        )
    if covered < len(probe):
        print(
            f"[preflight] WARNING: only {covered}/{len(probe)} probed videos have frames "
            f"in '{frames_path}'; some questions will run frame-starved.",
            flush=True,
        )


def evaluate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    frames = load_frames(config["frames_path"])
    examples = load_examples(config["qa_path"])
    if config.get("limit"):
        examples = examples[: int(config["limit"])]
    index = FrameIndex(frames)
    _preflight_frame_index(config, index, examples)
    answerer = build_answerer(config.get("answerer", {"kind": "lexical"}))
    output_dir = Path(config.get("output_dir", "runs/current"))
    ensure_dir(output_dir)
    ensure_dir(output_dir / "_shards")

    budgets = [int(x) for x in config.get("budgets", [4, 8, 16])]
    policy_names = list(config.get("policies", ["uniform", "mmr_retrieve", "scripted_curator", "best_single_frame"]))
    policy_config = dict(config.get("policy", {}))

    action_policy = None
    action_cfg = config.get("action_policy")
    if action_cfg:
        from .action_policy import build_action_policy

        action_policy = build_action_policy(
            model_path=str(action_cfg["model_path"]),
            adapter_path=str(action_cfg["adapter_path"]),
            max_new_tokens=int(action_cfg.get("max_new_tokens", 384)),
        )
        policy_config["_action_policy"] = action_policy

    # Crash-safe: compute each (budget, policy) block once, persist a shard, and
    # skip/reload completed shards so a killed job (e.g. SLURM wall-time) resumes
    # instead of losing everything (predictions were previously only written at the end).
    results: List[PolicyResult] = []
    for budget in budgets:
        for policy_name in policy_names:
            if policy_name == "best_single_frame" and budget != budgets[0]:
                continue
            shard = _shard_path(output_dir, budget, policy_name)
            cached = _load_completed_shard(shard, expected=len(examples))
            if cached is not None:
                print(f"[resume] {policy_name}@{budget}: loaded {len(cached)} cached results", flush=True)
                results.extend(cached)
                continue
            policy = build_policy(policy_name, index, answerer, budget, policy_config)
            block: List[PolicyResult] = []
            t_block_start = time.perf_counter()
            for example in examples:
                t0 = time.perf_counter()
                result = policy.run(example)
                latency_s = time.perf_counter() - t0
                # Inject latency into metadata without mutating the frozen dataclass
                meta = dict(result.metadata)
                meta["latency_s"] = round(latency_s, 4)
                result = PolicyResult(
                    qid=result.qid,
                    video_id=result.video_id,
                    policy=result.policy,
                    budget=result.budget,
                    answer=result.answer,
                    answer_index=result.answer_index,
                    confidence=result.confidence,
                    selected_frame_ids=result.selected_frame_ids,
                    selected_timestamps=result.selected_timestamps,
                    probes=result.probes,
                    num_retrievals=result.num_retrievals,
                    num_frames=result.num_frames,
                    trace=result.trace,
                    metadata=meta,
                )
                block.append(result)
            elapsed = time.perf_counter() - t_block_start
            write_jsonl(shard, [r.to_dict() for r in block])
            avg_latency = elapsed / len(block) if block else 0.0
            print(
                f"[done] {policy_name}@{budget}: {len(block)} results "
                f"avg_latency={avg_latency:.2f}s -> {shard}",
                flush=True,
            )
            results.extend(block)

    summary = summarize_results(results, examples)
    recall_ks = [int(k) for k in config.get("retrieval_recall_at_k", [4, 8, 16])]
    summary["retrieval_recall_at_k"] = {
        str(k): retrieval_recall_at_k(index, examples, k)
        for k in recall_ks
    }
    # Cost table: per (policy, budget) avg frames/tokens/latency
    cost_table: Dict[str, Any] = {}
    for budget in budgets:
        for policy_name in policy_names:
            key = f"{policy_name}@{budget}"
            block_results = [r for r in results if r.policy == policy_name and r.budget == budget]
            if not block_results:
                continue
            avg_frames = sum(r.num_frames for r in block_results) / len(block_results)
            avg_visual_tokens = avg_frames * 256  # low-detail estimate
            latencies = [r.metadata.get("latency_s") for r in block_results if r.metadata.get("latency_s") is not None]
            cost_table[key] = {
                "avg_frames": round(avg_frames, 3),
                "avg_visual_tokens_est": round(avg_visual_tokens, 1),
                "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            }
    summary["cost_table"] = cost_table
    summary["config"] = config

    shards_only = os.environ.get("EVAL_SHARDS_ONLY", "").lower() in ("1", "true", "yes")
    if shards_only:
        print("[shards-only] skipping summary.json (parallel shard worker)", flush=True)
        return summary

    save_results(output_dir / "predictions.jsonl", results)
    write_json(output_dir / "summary.json", summary)
    return summary


def finalize_from_shards(config: Dict[str, Any]) -> Dict[str, Any]:
    """Rebuild summary.json from completed _shards (after parallel policy workers)."""
    frames = load_frames(config["frames_path"])
    examples = load_examples(config["qa_path"])
    if config.get("limit"):
        examples = examples[: int(config["limit"])]
    index = FrameIndex(frames)
    output_dir = Path(config.get("output_dir", "runs/current"))
    budgets = [int(x) for x in config.get("budgets", [4, 8, 16])]
    policy_names = list(config.get("policies", []))

    results: List[PolicyResult] = []
    missing = []
    for budget in budgets:
        for policy_name in policy_names:
            if policy_name == "best_single_frame" and budget != budgets[0]:
                continue
            shard = _shard_path(output_dir, budget, policy_name)
            cached = _load_completed_shard(shard, expected=len(examples))
            if cached is None:
                missing.append(f"{policy_name}@{budget}")
                continue
            results.extend(cached)

    if missing:
        raise RuntimeError(f"cannot finalize {output_dir}: missing shards: {missing[:5]}{'...' if len(missing) > 5 else ''}")

    summary = summarize_results(results, examples)
    recall_ks = [int(k) for k in config.get("retrieval_recall_at_k", [4, 8, 16])]
    summary["retrieval_recall_at_k"] = {
        str(k): retrieval_recall_at_k(index, examples, k)
        for k in recall_ks
    }
    cost_table: Dict[str, Any] = {}
    for budget in budgets:
        for policy_name in policy_names:
            key = f"{policy_name}@{budget}"
            block_results = [r for r in results if r.policy == policy_name and r.budget == budget]
            if not block_results:
                continue
            avg_frames = sum(r.num_frames for r in block_results) / len(block_results)
            latencies = [r.metadata.get("latency_s") for r in block_results if r.metadata.get("latency_s") is not None]
            cost_table[key] = {
                "avg_frames": round(avg_frames, 3),
                "avg_visual_tokens_est": round(avg_frames * 256, 1),
                "avg_latency_s": round(sum(latencies) / len(latencies), 3) if latencies else None,
            }
    summary["cost_table"] = cost_table
    summary["config"] = config
    save_results(output_dir / "predictions.jsonl", results)
    write_json(output_dir / "summary.json", summary)
    return summary
