#!/usr/bin/env python3
"""Generate WACV figures from runs/*/summary.json.

Figures (PNG + PDF, under --out-dir, default reports/figures/):
  - pareto_<dataset>_<backbone>.png   accuracy vs frame budget, one line per policy
  - ce_auc_<dataset>_<backbone>.png   context-efficiency AUC bar chart
  - sfdelta_<dataset>_<backbone>.png  single-frame-shortcut gap vs budget
  - frames_vs_acc_<...>.png           efficiency frontier: avg_frames vs accuracy
  - calibration_<...>.png             PC-ECE per policy (lower = better calibrated)

Usage: python scripts/make_figures.py [--glob 'runs/gpu_*']
"""
from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# stable, distinguishable styling per policy
PLOT_ORDER = [
    "uniform", "mmr_retrieve", "aks", "videotree", "flexframe", "best_single_frame",
    "scripted_curator", "curator_siglip",
    "curator_probe_off", "curator_curation_off",
    "curator_summary_off", "curator_sufficiency_off", "curator_retrieval_off",
    "oracle_span", "curator_oracle_probe",
]


def parse_run_name(name: str):
    s = name[4:] if name.startswith("gpu_") else name
    backbone, split = "?", "?"
    for b in ["qwen2_5", "qwen3", "internvl3_5", "llava_video", "qwen3_thinking"]:
        if f"_{b}_" in f"_{s}_":
            backbone = b
            s = s.replace(f"_{b}", "")
            break
    for sp in ["full", "zero_shot", "test", "val"]:
        if s.endswith(f"_{sp}"):
            split = sp
            s = s[: -(len(sp) + 1)]
            break
    return s.strip("_"), backbone, split


def order_key(p):
    return PLOT_ORDER.index(p) if p in PLOT_ORDER else len(PLOT_ORDER)


def save(fig, base: Path):
    fig.savefig(f"{base}.png", dpi=150, bbox_inches="tight")
    fig.savefig(f"{base}.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="runs/gpu_*")
    ap.add_argument("--out-dir", default="reports/figures")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    n_fig = 0
    for sm in sorted(glob.glob(f"{args.glob}/summary.json")):
        run = Path(sm).parent.name
        ds, bb, sp = parse_run_name(run)
        tag = f"{ds}_{bb}_{sp}"
        try:
            s = json.load(open(sm))
        except Exception as exc:
            print(f"skip {run}: {exc}")
            continue

        # gather per-policy accuracy/budget/frames
        acc = defaultdict(dict)     # policy -> budget -> acc
        frames = defaultdict(dict)  # policy -> budget -> avg_frames
        ece = {}                    # policy -> pc_ece (at max budget seen)
        for key, g in s.get("groups", {}).items():
            pol, b = g["policy"], g["budget"]
            acc[pol][b] = g["accuracy"]
            frames[pol][b] = g.get("avg_frames", b)
            if g.get("pc_ece") is not None:
                ece[pol] = g["pc_ece"]

        # 1) Pareto: accuracy vs budget
        fig, ax = plt.subplots(figsize=(6, 4))
        for pol in sorted(acc, key=order_key):
            pts = sorted((b, a) for b, a in acc[pol].items() if b is not None)
            if len(pts) >= 1:
                xs, ys = zip(*pts)
                lw = 2.4 if pol == "scripted_curator" else 1.4
                ax.plot(xs, ys, marker="o", linewidth=lw, label=pol)
        ax.set_xlabel("frame budget K"); ax.set_ylabel("accuracy")
        ax.set_title(f"Accuracy vs budget — {tag}")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        save(fig, out / f"pareto_{tag}"); n_fig += 1

        # 2) frames vs accuracy efficiency frontier (best budget per policy)
        fig, ax = plt.subplots(figsize=(6, 4))
        for pol in sorted(acc, key=order_key):
            xs = [frames[pol][b] for b in sorted(acc[pol]) if b in frames[pol]]
            ys = [acc[pol][b] for b in sorted(acc[pol]) if b in frames[pol]]
            if xs:
                ax.plot(xs, ys, marker="s", label=pol,
                        linewidth=2.4 if pol == "scripted_curator" else 1.2)
        ax.set_xlabel("avg frames used"); ax.set_ylabel("accuracy")
        ax.set_title(f"Efficiency frontier — {tag}")
        ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
        save(fig, out / f"frames_vs_acc_{tag}"); n_fig += 1

        # 3) CE-AUC bars
        ce = {k: v for k, v in (s.get("context_efficiency") or {}).items() if v is not None}
        if ce:
            fig, ax = plt.subplots(figsize=(6, 4))
            pols = sorted(ce, key=order_key)
            ax.bar(range(len(pols)), [ce[p] for p in pols],
                   color=["#d62728" if p == "scripted_curator" else "#1f77b4" for p in pols])
            ax.set_xticks(range(len(pols))); ax.set_xticklabels(pols, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("CE-AUC"); ax.set_title(f"Context efficiency — {tag}")
            ax.grid(alpha=0.3, axis="y")
            save(fig, out / f"ce_auc_{tag}"); n_fig += 1

        # 4) SF-delta vs budget
        sfd = s.get("sf_delta") or {}
        if sfd:
            by_pol = defaultdict(dict)
            for k, v in sfd.items():
                if "@" in k:
                    p, b = k.rsplit("@", 1)
                    by_pol[p][int(b)] = v
            fig, ax = plt.subplots(figsize=(6, 4))
            for pol in sorted(by_pol, key=order_key):
                pts = sorted(by_pol[pol].items())
                xs, ys = zip(*pts)
                ax.plot(xs, ys, marker="o", label=pol,
                        linewidth=2.4 if pol == "scripted_curator" else 1.2)
            ax.axhline(0, color="gray", lw=0.8, ls="--")
            ax.set_xlabel("frame budget K"); ax.set_ylabel("SF-Δ (multi - single frame)")
            ax.set_title(f"Temporal evidence (SF-Δ) — {tag}")
            ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
            save(fig, out / f"sfdelta_{tag}"); n_fig += 1

        # 5) calibration (PC-ECE) bars
        if ece:
            fig, ax = plt.subplots(figsize=(6, 4))
            pols = sorted(ece, key=order_key)
            ax.bar(range(len(pols)), [ece[p] for p in pols],
                   color=["#d62728" if p == "scripted_curator" else "#2ca02c" for p in pols])
            ax.set_xticks(range(len(pols))); ax.set_xticklabels(pols, rotation=45, ha="right", fontsize=7)
            ax.set_ylabel("PC-ECE (lower=better)"); ax.set_title(f"Calibration — {tag}")
            ax.grid(alpha=0.3, axis="y")
            save(fig, out / f"calibration_{tag}"); n_fig += 1

        # 6) Risk-coverage curves (abstention) — one line per policy at max budget
        risk_cov = s.get("risk_coverage") or {}
        if risk_cov:
            fig, ax = plt.subplots(figsize=(6, 4))
            plotted = set()
            for key, curve in sorted(risk_cov.items()):
                pol = key.rsplit("@", 1)[0] if "@" in key else key
                if pol in plotted or not curve:
                    continue
                # Only show the max budget per policy to reduce clutter
                all_keys = [k for k in risk_cov if k.startswith(pol + "@")]
                max_key = max(all_keys, key=lambda k: int(k.rsplit("@", 1)[1]) if k.rsplit("@", 1)[1].isdigit() else 0)
                if key != max_key:
                    continue
                xs = [pt["coverage"] for pt in curve]
                ys = [pt["accuracy"] for pt in curve]
                lw = 2.4 if pol == "scripted_curator" else 1.4
                ax.plot(xs, ys, marker="o", linewidth=lw, label=pol)
                plotted.add(pol)
            ax.set_xlabel("coverage (fraction answered)"); ax.set_ylabel("accuracy of answered subset")
            ax.set_title(f"Risk-coverage (abstention) — {tag}")
            ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3)
            save(fig, out / f"risk_coverage_{tag}"); n_fig += 1

        # 7) SF-Δ by question type (heatmap style — grouped bars per domain)
        sf_by_type = s.get("sf_delta_by_type") or {}
        if sf_by_type:
            # Collect (policy, budget) -> domain -> delta for the max budget per policy
            type_data: dict = defaultdict(dict)
            for pb_key, type_map in sf_by_type.items():
                pol = pb_key.rsplit("@", 1)[0] if "@" in pb_key else pb_key
                for domain, delta in (type_map or {}).items():
                    if delta is not None:
                        type_data[pol][domain] = delta
            if type_data:
                all_domains = sorted({d for v in type_data.values() for d in v})
                pols_typed = sorted(type_data, key=order_key)
                x = range(len(all_domains))
                width = 0.8 / max(1, len(pols_typed))
                fig, ax = plt.subplots(figsize=(max(6, len(all_domains) * 1.2), 4))
                for i, pol in enumerate(pols_typed):
                    vals = [type_data[pol].get(d, 0.0) for d in all_domains]
                    offsets = [xi + (i - len(pols_typed) / 2 + 0.5) * width for xi in x]
                    ax.bar(offsets, vals, width=width * 0.9, label=pol,
                           color="#d62728" if pol == "scripted_curator" else None)
                ax.axhline(0, color="gray", lw=0.8, ls="--")
                ax.set_xticks(list(x)); ax.set_xticklabels(all_domains, rotation=40, ha="right", fontsize=7)
                ax.set_ylabel("SF-Δ per question type"); ax.set_title(f"SF-Δ by type — {tag}")
                ax.legend(fontsize=7, ncol=2); ax.grid(alpha=0.3, axis="y")
                save(fig, out / f"sfdelta_by_type_{tag}"); n_fig += 1

    print(f"wrote {n_fig} figures (png+pdf) to {out}/")
    return 0 if n_fig else 1


if __name__ == "__main__":
    raise SystemExit(main())
