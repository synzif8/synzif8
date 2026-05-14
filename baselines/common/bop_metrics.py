"""BOP-style aggregate metrics + per-category report formatter.

Inputs are per-instance records produced by `compute_metrics` in
overfit_runner_ddp.py:
    {scene_id, obj_id, add_s, r_err, t_err, [mpjpe]}

Adds:
  - ADD-S 0.1d: pass-rate where add_s ≤ 0.1 × diameter
  - AUC of ADD: integral of recall(threshold) over [0, 0.1d] × 100 bins
  - format_results_table: per-category markdown table renderer
"""

from __future__ import annotations

import json
import os
from typing import Iterable

import numpy as np


DEFAULT_MODELS_INFO = "<PROJECT_ROOT>/baselines/phase0_C/models/models_info.json"


def load_diameter(models_info_path: str = DEFAULT_MODELS_INFO, obj_key: str = "1") -> float:
    """Load object diameter from BOP-style models_info.json."""
    if not os.path.exists(models_info_path):
        # Fallback: rhombic dodecahedron canonical diameter ≈ 4 (axial range 4),
        # but with corner vertices at distance √3 ≈ 1.73 the bbox diagonal is √(4²+4²+4²) ≈ 6.93.
        return 6.928203230275509
    with open(models_info_path) as f:
        info = json.load(f)
    return float(info[obj_key]["diameter"])


def add_s_pass_rate(records: Iterable[dict], diameter: float, frac: float = 0.1) -> float:
    """Fraction of instances with add_s ≤ frac × diameter."""
    arr = np.asarray([r["add_s"] for r in records], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    thresh = frac * diameter
    return float((arr <= thresh).mean())


def auc_add(records: Iterable[dict], diameter: float, max_frac: float = 0.1, n_bins: int = 100) -> float:
    """Area under recall(threshold) for threshold ∈ [0, max_frac × diameter], normalised to [0, 1]."""
    arr = np.asarray([r["add_s"] for r in records], dtype=np.float64)
    if arr.size == 0:
        return float("nan")
    max_t = max_frac * diameter
    thresholds = np.linspace(0.0, max_t, n_bins + 1)
    recalls = np.asarray([(arr <= t).mean() for t in thresholds])
    # trapezoidal area / max_t (so AUC is in [0, 1])
    return float(np.trapz(recalls, thresholds) / max_t)


def aggregate_with_bop(records: list[dict], diameter: float) -> dict:
    """Mean of per-instance metrics + ADD-S 0.1d + AUC."""
    if not records:
        return {}
    keys = [k for k in records[0] if k not in ("scene_id", "obj_id")]
    out = {k: float(np.mean([r[k] for r in records])) for k in keys}
    out["add_s_0_1d"] = add_s_pass_rate(records, diameter, frac=0.1)
    out["auc_add"] = auc_add(records, diameter, max_frac=0.1, n_bins=100)
    return out


CATEGORY_METRIC_KEYS = {
    "A": ["add_s", "add_s_0_1d", "auc_add", "r_err", "t_err"],
    "B": ["mpjpe", "add_s", "add_s_0_1d", "auc_add", "r_err", "t_err"],
    "C": ["add_s", "add_s_0_1d", "auc_add", "r_err", "t_err"],
}

PRETTY_HEADER = {
    "add_s": "ADD-S",
    "add_s_0_1d": "ADD-S 0.1d",
    "auc_add": "AUC ADD",
    "r_err": "R err°",
    "t_err": "t err",
    "mpjpe": "MPJPE",
}


def format_table(category: str, rows: list[dict]) -> str:
    """rows: [{model, mode, <metric_key>...}, ...]. Returns markdown table string."""
    keys = CATEGORY_METRIC_KEYS[category]
    headers = ["Model", "Mode"] + [PRETTY_HEADER[k] for k in keys]
    sep = ["---"] * len(headers)
    out = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for r in rows:
        cells = [str(r.get("model", "")), str(r.get("mode", "")) ]
        for k in keys:
            v = r.get(k)
            if v is None or (isinstance(v, float) and (v != v)):  # NaN
                cells.append("—")
            else:
                cells.append(f"{v:.4f}" if isinstance(v, float) else str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_results_md(per_category_rows: dict[str, list[dict]], path_md: str) -> None:
    title = "# Phase-1 Test Set Quantitative Evaluation"
    blocks = [title, ""]
    for cat in ("A", "B", "C"):
        blocks.append(f"## Category {cat}")
        rows = per_category_rows.get(cat, [])
        blocks.append(format_table(cat, rows) if rows else "_no rows_")
        blocks.append("")
    with open(path_md, "w") as f:
        f.write("\n".join(blocks) + "\n")


def write_results_csv(per_category_rows: dict[str, list[dict]], path_csv: str) -> None:
    import csv as _csv
    fieldnames = ["category", "model", "mode", "add_s", "add_s_0_1d", "auc_add", "r_err", "t_err", "mpjpe"]
    with open(path_csv, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for cat, rows in per_category_rows.items():
            for r in rows:
                row = {"category": cat}
                for k in fieldnames[1:]:
                    row[k] = r.get(k, "")
                w.writerow(row)
