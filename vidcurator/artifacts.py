"""Paper artifact export helpers."""

from __future__ import annotations

import csv
import html
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .io import ensure_dir, read_json, write_json


def export_run_artifacts(run_dir: str | Path) -> Dict[str, str]:
    run_path = Path(run_dir)
    summary_path = run_path / "summary.json"
    summary = read_json(summary_path)
    tables_dir = run_path / "tables"
    figures_dir = run_path / "figures"
    ensure_dir(tables_dir)
    ensure_dir(figures_dir)

    outputs: Dict[str, str] = {}
    outputs["metrics_csv"] = str(_write_metrics_csv(summary, tables_dir / "metrics_by_policy_budget.csv"))
    outputs["ce_csv"] = str(_write_context_efficiency_csv(summary, tables_dir / "context_efficiency.csv"))
    outputs["sf_csv"] = str(_write_sf_delta_csv(summary, tables_dir / "single_frame_shortcut_delta.csv"))
    outputs["accuracy_svg"] = str(_write_accuracy_svg(summary, figures_dir / "accuracy_vs_budget.svg"))
    outputs["fep_svg"] = str(_write_metric_bar_svg(summary, figures_dir / "frame_evidence_precision.svg", "fep"))
    outputs["sf_svg"] = str(_write_sf_delta_svg(summary, figures_dir / "single_frame_shortcut_delta.svg"))
    write_json(run_path / "artifact_manifest.json", outputs)
    return outputs


def _write_metrics_csv(summary: Dict[str, Any], path: Path) -> Path:
    fields = [
        "policy",
        "budget",
        "n",
        "accuracy",
        "avg_frames",
        "avg_retrievals",
        "fep",
        "fer",
        "if",
        "pc_ece",
    ]
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary.get("groups", {}).values():
            writer.writerow({field: row.get(field) for field in fields})
    return path


def _write_context_efficiency_csv(summary: Dict[str, Any], path: Path) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy", "ce_auc"])
        writer.writeheader()
        for policy, value in sorted(summary.get("context_efficiency", {}).items()):
            writer.writerow({"policy": policy, "ce_auc": value})
    return path


def _write_sf_delta_csv(summary: Dict[str, Any], path: Path) -> Path:
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["policy_budget", "sf_delta"])
        writer.writeheader()
        for key, value in sorted(summary.get("sf_delta", {}).items()):
            writer.writerow({"policy_budget": key, "sf_delta": value})
    return path


def _group_points(summary: Dict[str, Any], metric: str) -> Dict[str, List[Tuple[int, float]]]:
    points: Dict[str, List[Tuple[int, float]]] = {}
    for row in summary.get("groups", {}).values():
        value = row.get(metric)
        if value is None:
            continue
        points.setdefault(row["policy"], []).append((int(row["budget"]), float(value)))
    for policy in points:
        points[policy].sort()
    return points


def _write_accuracy_svg(summary: Dict[str, Any], path: Path) -> Path:
    return _write_line_svg(
        _group_points(summary, "accuracy"),
        path,
        title="Accuracy vs Frame Budget",
        y_label="accuracy",
    )


def _write_line_svg(points: Dict[str, List[Tuple[int, float]]], path: Path, title: str, y_label: str) -> Path:
    width, height = 760, 420
    left, right, top, bottom = 70, 30, 45, 65
    all_points = [point for series in points.values() for point in series]
    if not all_points:
        path.write_text("<svg xmlns='http://www.w3.org/2000/svg'></svg>\n", encoding="utf-8")
        return path
    min_x = min(x for x, _ in all_points)
    max_x = max(x for x, _ in all_points)
    min_y = 0.0
    max_y = max(1.0, max(y for _, y in all_points))

    def sx(x: float) -> float:
        if max_x == min_x:
            return left
        return left + (x - min_x) / (max_x - min_x) * (width - left - right)

    def sy(y: float) -> float:
        return height - bottom - (y - min_y) / (max_y - min_y) * (height - top - bottom)

    colors = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#d97706", "#0891b2"]
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='26' text-anchor='middle' font-family='Arial' font-size='18'>{html.escape(title)}</text>",
        f"<line x1='{left}' y1='{top}' x2='{left}' y2='{height-bottom}' stroke='#111827'/>",
        f"<line x1='{left}' y1='{height-bottom}' x2='{width-right}' y2='{height-bottom}' stroke='#111827'/>",
        f"<text x='{width/2}' y='{height-18}' text-anchor='middle' font-family='Arial' font-size='13'>frames in context</text>",
        f"<text x='18' y='{height/2}' transform='rotate(-90 18 {height/2})' text-anchor='middle' font-family='Arial' font-size='13'>{html.escape(y_label)}</text>",
    ]
    for idx, (policy, series) in enumerate(sorted(points.items())):
        color = colors[idx % len(colors)]
        path_data = " ".join(f"{'M' if i == 0 else 'L'} {sx(x):.1f} {sy(y):.1f}" for i, (x, y) in enumerate(series))
        parts.append(f"<path d='{path_data}' fill='none' stroke='{color}' stroke-width='2.5'/>")
        for x, y in series:
            parts.append(f"<circle cx='{sx(x):.1f}' cy='{sy(y):.1f}' r='4' fill='{color}'/>")
        legend_y = 52 + idx * 20
        parts.append(f"<rect x='{width-210}' y='{legend_y-10}' width='12' height='12' fill='{color}'/>")
        parts.append(
            f"<text x='{width-192}' y='{legend_y}' font-family='Arial' font-size='12'>{html.escape(policy)}</text>"
        )
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path


def _write_metric_bar_svg(summary: Dict[str, Any], path: Path, metric: str) -> Path:
    values = []
    for key, row in sorted(summary.get("groups", {}).items()):
        value = row.get(metric)
        if value is not None:
            values.append((key, float(value)))
    return _write_bar_svg(values, path, title=metric.upper())


def _write_sf_delta_svg(summary: Dict[str, Any], path: Path) -> Path:
    values = [(key, float(value)) for key, value in sorted(summary.get("sf_delta", {}).items())]
    return _write_bar_svg(values, path, title="Single-Frame Shortcut Delta")


def _write_bar_svg(values: Iterable[Tuple[str, float]], path: Path, title: str) -> Path:
    rows = list(values)
    width = 820
    bar_h = 22
    height = max(160, 70 + len(rows) * 34)
    left = 260
    right = 40
    max_abs = max([abs(v) for _, v in rows] + [1.0])
    zero_x = left + (width - left - right) / 2
    parts = [
        f"<svg xmlns='http://www.w3.org/2000/svg' width='{width}' height='{height}' viewBox='0 0 {width} {height}'>",
        "<rect width='100%' height='100%' fill='white'/>",
        f"<text x='{width/2}' y='28' text-anchor='middle' font-family='Arial' font-size='18'>{html.escape(title)}</text>",
        f"<line x1='{zero_x:.1f}' y1='50' x2='{zero_x:.1f}' y2='{height-25}' stroke='#6b7280'/>",
    ]
    for idx, (label, value) in enumerate(rows):
        y = 58 + idx * 34
        length = (abs(value) / max_abs) * ((width - left - right) / 2)
        x = zero_x if value >= 0 else zero_x - length
        color = "#059669" if value >= 0 else "#dc2626"
        parts.append(f"<text x='12' y='{y+15}' font-family='Arial' font-size='12'>{html.escape(label)}</text>")
        parts.append(f"<rect x='{x:.1f}' y='{y}' width='{length:.1f}' height='{bar_h}' fill='{color}'/>")
        parts.append(f"<text x='{x + length + 6:.1f}' y='{y+15}' font-family='Arial' font-size='12'>{value:.3f}</text>")
    parts.append("</svg>")
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")
    return path
