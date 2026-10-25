"""Evaluation metrics for VID-CURATOR."""

from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .retrieval import timestamp_hits_spans, token_set
from .types import PolicyResult, QAExample

NORM_RE = re.compile(r"[^a-z0-9]+")


def normalize_text(text: str) -> str:
    return NORM_RE.sub(" ", text.lower()).strip()


def is_correct(result: PolicyResult, example: QAExample) -> bool:
    if example.answer_index is not None and result.answer_index is not None:
        return int(example.answer_index) == int(result.answer_index)
    gold = normalize_text(example.correct_answer_text())
    pred = normalize_text(result.answer)
    return bool(gold and pred and (gold == pred or gold in pred or pred in gold))


def wilson_ci(acc: float, n: int, z: float = 1.96) -> Tuple[Optional[float], Optional[float]]:
    """95% Wilson score interval."""
    if not n:
        return None, None
    denom = 1 + z * z / n
    center = (acc + z * z / (2 * n)) / denom
    half = (z * math.sqrt((acc * (1 - acc) + z * z / (4 * n)) / n)) / denom
    return round(max(0.0, center - half), 4), round(min(1.0, center + half), 4)


def frame_evidence_pr(
    selected_timestamps: Sequence[float],
    example: QAExample,
    tolerance: float = 1.0,
) -> Tuple[Optional[float], Optional[float]]:
    if not example.evidence_spans:
        return None, None
    if not selected_timestamps:
        return 0.0, 0.0
    hits = [timestamp_hits_spans(ts, example.evidence_spans, tolerance=tolerance) for ts in selected_timestamps]
    precision = sum(1 for hit in hits if hit) / len(hits)
    recall = 1.0 if any(hits) else 0.0
    return precision, recall


def introspection_hit(result: PolicyResult, example: QAExample) -> Optional[float]:
    if not example.evidence_spans or not result.probes:
        return None
    target_tokens = token_set(" ".join(span.label for span in example.evidence_spans))
    if not target_tokens:
        if any(timestamp_hits_spans(ts, example.evidence_spans, tolerance=2.0) for ts in result.selected_timestamps):
            return 1.0
        return 0.0
    probe_text = " ".join(str(probe.get("missing_moment", "")) for probe in result.probes)
    return 1.0 if token_set(probe_text) & target_tokens else 0.0


def calibration_bins(pairs: Sequence[Tuple[float, bool]], bins: int = 10) -> Dict[str, Any]:
    if not pairs:
        return {"ece": None, "bins": []}
    bucket_rows: List[Dict[str, Any]] = []
    ece = 0.0
    n = len(pairs)
    for bucket in range(bins):
        lo = bucket / bins
        hi = (bucket + 1) / bins
        rows = [(conf, corr) for conf, corr in pairs if lo <= conf < hi or (bucket == bins - 1 and conf == 1.0)]
        if not rows:
            continue
        avg_conf = sum(conf for conf, _ in rows) / len(rows)
        acc = sum(1 for _, corr in rows if corr) / len(rows)
        ece += (len(rows) / n) * abs(avg_conf - acc)
        bucket_rows.append({"lo": lo, "hi": hi, "count": len(rows), "avg_conf": avg_conf, "accuracy": acc})
    return {"ece": ece, "bins": bucket_rows}


def context_efficiency_auc(points: Sequence[Tuple[int, float]]) -> Optional[float]:
    if len(points) < 2:
        return None
    ordered = sorted(points)
    min_k = ordered[0][0]
    max_k = ordered[-1][0]
    if max_k == min_k:
        return None
    area = 0.0
    for (k1, a1), (k2, a2) in zip(ordered, ordered[1:]):
        area += (k2 - k1) * (a1 + a2) / 2.0
    return area / (max_k - min_k)


def risk_coverage_curve(
    pairs: Sequence[Tuple[float, bool]], steps: int = 10
) -> List[Dict[str, float]]:
    """Compute risk-coverage (abstention) curve.

    Sort predictions by descending confidence, then sweep a threshold from 0→1
    to produce (coverage, accuracy) pairs.  High-confidence subset = low risk.
    Returns a list of dicts with keys: threshold, coverage, accuracy.
    """
    if not pairs:
        return []
    sorted_pairs = sorted(pairs, key=lambda x: x[0], reverse=True)
    n = len(sorted_pairs)
    thresholds = [i / steps for i in range(steps + 1)]
    curve = []
    for threshold in thresholds:
        subset = [(conf, ok) for conf, ok in sorted_pairs if conf >= threshold]
        coverage = len(subset) / n
        accuracy = sum(1 for _, ok in subset if ok) / len(subset) if subset else 0.0
        curve.append({"threshold": round(threshold, 4), "coverage": round(coverage, 4), "accuracy": round(accuracy, 4)})
    return curve


def sf_delta_by_question_type(
    correctness: Dict[str, bool],
    single_correctness: Dict[str, bool],
    examples: Sequence[QAExample],
) -> Dict[str, Optional[float]]:
    """Compute SF-Δ broken down by question domain/type."""
    by_type: Dict[str, Tuple[List[bool], List[bool]]] = defaultdict(lambda: ([], []))
    for example in examples:
        qid = example.qid
        if qid not in correctness or qid not in single_correctness:
            continue
        by_type[example.domain][0].append(correctness[qid])
        by_type[example.domain][1].append(single_correctness[qid])
    result: Dict[str, Optional[float]] = {}
    for domain, (full_oks, single_oks) in sorted(by_type.items()):
        if not full_oks:
            result[domain] = None
            continue
        result[domain] = round(
            sum(full_oks) / len(full_oks) - sum(single_oks) / len(single_oks), 4
        )
    return result


def cost_metrics(rows: Sequence[PolicyResult]) -> Dict[str, Any]:
    """Estimate cost metrics for a policy run.

    Tracks: avg_frames (actual), avg_probes, avg_retrievals.
    Visual-token estimate: each frame ~256 tokens (low-detail 28*28 tile).
    """
    if not rows:
        return {}
    n = len(rows)
    avg_frames = sum(r.num_frames for r in rows) / n
    avg_retrievals = sum(r.num_retrievals for r in rows) / n
    avg_probes = sum(len(r.probes) for r in rows) / n
    tokens_per_frame = 256  # low-detail 28×28 patch estimate
    avg_visual_tokens = avg_frames * tokens_per_frame
    latencies = [r.metadata.get("latency_s", None) for r in rows]
    latencies_valid = [lt for lt in latencies if lt is not None]
    return {
        "avg_frames": round(avg_frames, 3),
        "avg_retrievals": round(avg_retrievals, 3),
        "avg_probes": round(avg_probes, 3),
        "avg_visual_tokens_est": round(avg_visual_tokens, 1),
        "avg_latency_s": round(sum(latencies_valid) / len(latencies_valid), 3) if latencies_valid else None,
    }


def summarize_results(results: Sequence[PolicyResult], examples: Sequence[QAExample]) -> Dict[str, Any]:
    by_qid = {example.qid: example for example in examples}
    groups: Dict[Tuple[str, int], List[PolicyResult]] = defaultdict(list)
    for result in results:
        groups[(result.policy, result.budget)].append(result)

    summary: Dict[str, Any] = {
        "groups": {},
        "context_efficiency": {},
        "sf_delta": {},
        "sf_delta_by_type": {},
        "risk_coverage": {},
    }
    acc_points: Dict[str, List[Tuple[int, float]]] = defaultdict(list)
    correctness_by_group: Dict[Tuple[str, int], Dict[str, bool]] = {}

    for (policy, budget), rows in sorted(groups.items()):
        correctness = [is_correct(result, by_qid[result.qid]) for result in rows]
        correctness_by_group[(policy, budget)] = {result.qid: ok for result, ok in zip(rows, correctness)}
        fep_values: List[float] = []
        fer_values: List[float] = []
        if_values: List[float] = []
        cal_pairs: List[Tuple[float, bool]] = []
        domains: Dict[str, List[bool]] = defaultdict(list)
        for result, ok in zip(rows, correctness):
            example = by_qid[result.qid]
            fep, fer = frame_evidence_pr(result.selected_timestamps, example)
            if fep is not None:
                fep_values.append(fep)
            if fer is not None:
                fer_values.append(fer)
            hit = introspection_hit(result, example)
            if hit is not None:
                if_values.append(hit)
            cal_pairs.append((result.confidence, ok))
            domains[example.domain].append(ok)
        acc = sum(1 for ok in correctness if ok) / len(correctness) if correctness else 0.0
        n = len(correctness)
        ci_lo, ci_hi = wilson_ci(acc, n)
        acc_points[policy].append((budget, acc))
        group_key = f"{policy}@{budget}"
        summary["groups"][group_key] = {
            "policy": policy,
            "budget": budget,
            "n": n,
            "accuracy": acc,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "avg_frames": sum(result.num_frames for result in rows) / len(rows) if rows else 0.0,
            "avg_retrievals": sum(result.num_retrievals for result in rows) / len(rows) if rows else 0.0,
            "avg_probes": sum(len(result.probes) for result in rows) / len(rows) if rows else 0.0,
            "avg_visual_tokens_est": sum(result.num_frames for result in rows) / len(rows) * 256 if rows else 0.0,
            "fep": _mean(fep_values),
            "fer": _mean(fer_values),
            "if": _mean(if_values),
            "pc_ece": calibration_bins(cal_pairs)["ece"],
            "risk_coverage": risk_coverage_curve(cal_pairs),
            "domains": {
                domain: sum(1 for ok in oks if ok) / len(oks)
                for domain, oks in sorted(domains.items())
                if oks
            },
        }
        summary["risk_coverage"][group_key] = risk_coverage_curve(cal_pairs)

    for policy, points in sorted(acc_points.items()):
        summary["context_efficiency"][policy] = context_efficiency_auc(points)

    budgets = sorted({result.budget for result in results if result.policy != "best_single_frame"})
    single_rows = [key for key in correctness_by_group if key[0] == "best_single_frame"]
    for policy in sorted({result.policy for result in results if result.policy != "best_single_frame"}):
        for budget in budgets:
            key = (policy, budget)
            if key not in correctness_by_group or not single_rows:
                continue
            single_key = single_rows[0]
            qids = set(correctness_by_group[key]) & set(correctness_by_group[single_key])
            if not qids:
                continue
            full_acc = sum(1 for qid in qids if correctness_by_group[key][qid]) / len(qids)
            single_acc = sum(1 for qid in qids if correctness_by_group[single_key][qid]) / len(qids)
            sfd_key = f"{policy}@{budget}"
            summary["sf_delta"][sfd_key] = full_acc - single_acc
            # Per question-type SF-Δ
            type_delta = sf_delta_by_question_type(
                correctness_by_group[key],
                correctness_by_group[single_key],
                [by_qid[qid] for qid in qids if qid in by_qid],
            )
            summary["sf_delta_by_type"][sfd_key] = type_delta

    return summary


def _mean(values: Sequence[float]) -> Optional[float]:
    if not values:
        return None
    finite = [v for v in values if not math.isnan(v)]
    return sum(finite) / len(finite) if finite else None
