#!/usr/bin/env python3
"""RFT: build an outcome-reward-optimal SFT dataset (offline, no verl, no GPU).

The cold-start SFT distilled the SCRIPTED curator's correct traces. That caps the policy
at the scripted reference (hence the flat GRPO reward). RFT instead distills, per question,
the trajectory that MAXIMISES an OUTCOME reward (compute_reward: correct − frame cost +
evidence + calibration) chosen across ALL policies × budgets we already ran (uniform, aks,
scripted, curator, learned). This is Best-of-N rejection fine-tuning / reward-weighted policy
improvement — done entirely offline from saved predictions, then trained with the working
LoRA SFT (scripts/sft_train.py). Novel for a curation policy; robust (bypasses fragile verl).

Reward variants (see REWARD_PRESETS): pure_outcome, acc_per_frame, evidence, efficient_grounded,
calibrated. Only policies with action traces (scripted/curator/learned) can be distilled.

Usage: python scripts/rft_build_dataset.py --reward acc_per_frame \
         --datasets nextgqa --pred-runs r1_v2_%s_qwen2_5 r3_v2_%s_learned \
         --out data/rl/rft_acc_per_frame_sft.jsonl
"""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

from vidcurator.io import load_examples
from vidcurator.types import PolicyResult
from vidcurator.grpo import compute_reward, extract_sft_traces, RewardWeights
from vidcurator.metrics import is_correct

# Reward variants — weights over compute_reward components.
REWARD_PRESETS = {
    # pure downstream accuracy (what verl imitation reward never optimised)
    "pure_outcome":       {"correct": 1.0, "num_frames": 0.0, "num_retrievals": 0.0, "num_probes": 0.0,
                           "frame_evidence_precision": 0.0, "calibrated_stop": 0.0, "format_valid": 0.1},
    # accuracy-per-frame — directly optimises the paper's CE-AUC / efficiency axis
    "acc_per_frame":      {"correct": 1.0, "num_frames": 0.5, "num_retrievals": 0.1, "num_probes": 0.1,
                           "frame_evidence_precision": 0.0, "calibrated_stop": 0.0, "format_valid": 0.1},
    # evidence-grounded — reward selecting gold-evidence frames (ties to E-A)
    "evidence":           {"correct": 1.0, "num_frames": 0.1, "num_retrievals": 0.05, "num_probes": 0.05,
                           "frame_evidence_precision": 0.6, "calibrated_stop": 0.0, "format_valid": 0.1},
    # combined novelty target — efficient AND grounded AND calibrated
    "efficient_grounded": {"correct": 1.0, "num_frames": 0.3, "num_retrievals": 0.05, "num_probes": 0.05,
                           "frame_evidence_precision": 0.4, "calibrated_stop": 0.2, "format_valid": 0.1},
    # calibration-shaped — reward confidence tracking correctness (probes E-C's dead pillar)
    "calibrated":         {"correct": 1.0, "num_frames": 0.1, "num_retrievals": 0.05, "num_probes": 0.05,
                           "frame_evidence_precision": 0.0, "calibrated_stop": 0.5, "format_valid": 0.1},
}
# Only these policies carry probe/retrieve action traces distillable by extract_sft_traces.
DISTILLABLE = {"scripted_curator", "curator_siglip", "learned_curator"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reward", required=True, choices=list(REWARD_PRESETS))
    ap.add_argument("--datasets", nargs="+", default=["nextgqa"])
    ap.add_argument("--pred-runs", nargs="+", default=["r1_v2_%s_qwen2_5", "r3_v2_%s_learned"],
                    help="run-dir templates (%%s = dataset) to pool trajectories from")
    ap.add_argument("--qa-template", default="data/processed/%s_val_150.jsonl")
    ap.add_argument("--qa-overrides", nargs="*", default=["nextqa=data/processed/nextqa_temporal_val_150.jsonl"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    weights = RewardWeights.from_dict(REWARD_PRESETS[args.reward])
    qa_over = dict(o.split("=", 1) for o in args.qa_overrides)

    n_written = 0
    stats = {"qids": 0, "avg_reward": 0.0, "avg_frames": 0.0, "policy_mix": defaultdict(int)}
    out = Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fout = open(out, "w", encoding="utf-8")
    for ds in args.datasets:
        qa_path = qa_over.get(ds, args.qa_template % ds)
        examples = {e.qid: e for e in load_examples(qa_path)}
        # pool all candidate trajectories per qid
        cand = defaultdict(list)   # qid -> [PolicyResult]
        for tmpl in args.pred_runs:
            pf = Path("runs") / (tmpl % ds) / "predictions.jsonl"
            if not pf.exists():
                print(f"  … {pf} missing, skip"); continue
            for line in open(pf):
                r = json.loads(line)
                if r.get("policy") not in DISTILLABLE:
                    continue
                try:
                    res = PolicyResult.from_dict(r)
                except Exception:
                    continue
                cand[res.qid].append(res)
        # pick best-reward correct trajectory per qid
        for qid, results in cand.items():
            ex = examples.get(qid)
            if ex is None:
                continue
            scored = []
            for res in results:
                if not is_correct(res, ex):
                    continue  # never distil a wrong trajectory
                rd = compute_reward(res, ex, weights=weights)
                scored.append((rd["reward"], res))
            if not scored:
                continue
            scored.sort(key=lambda x: x[0], reverse=True)
            best_reward, best = scored[0]
            traces = extract_sft_traces(best, ex, only_correct=True)
            if not traces:
                continue
            for t in traces:
                fout.write(json.dumps(t, ensure_ascii=False) + "\n")
                n_written += 1
            stats["qids"] += 1
            stats["avg_reward"] += best_reward
            stats["avg_frames"] += best.num_frames
            stats["policy_mix"][best.policy] += 1
    fout.close()
    q = stats["qids"] or 1
    print(f"reward={args.reward}: {stats['qids']} qids, {n_written} SFT turns → {out}")
    print(f"  avg_selected_reward={stats['avg_reward']/q:.4f}  avg_selected_frames={stats['avg_frames']/q:.2f}")
    print(f"  selected-policy mix: {dict(stats['policy_mix'])}")
    return 0 if n_written else 1


if __name__ == "__main__":
    raise SystemExit(main())
