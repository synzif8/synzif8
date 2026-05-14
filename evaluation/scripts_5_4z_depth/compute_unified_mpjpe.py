"""Step 1 — Compute unified MPJPE (sym-aware mean per-vertex L2) for all 9 baselines.

Unifies ADD-S (A/C) and MPJPE (B) under a single metric name "MPJPE",
computed identically as sym-aware mean per-vertex L2 over O_24 group.

Per instance:
    MPJPE_i = min over G in O_24 of mean_v ||model_verts_v - (s_gt c_v G^T) R_gt^T - t_gt||_2

  Input vertices per category:
    A/C : pred_verts (reconstructed from R_pred, t_pred, s_pred)
    B   : kp3d_raw   (model's raw 14-vertex output, no Procrustes)

Aggregate over n=11,379 instances per model. mean ± std, 4dp.

Outputs:
  - stdout : per-model table + comparison with existing ADD-S/MPJPE values
  - results/5.4Z_depth/unified_mpjpe_summary.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("<PROJECT_ROOT>/evaluation")
PER_MODEL = ROOT / "results" / "per_model"
OUT_CSV = ROOT / "results" / "5.4Z_depth" / "unified_mpjpe_summary.csv"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "<PROJECT_ROOT>")
from edge_utils import get_canonical_vertices  # type: ignore  # noqa: E402
from lineup.common import geometry as geo       # type: ignore  # noqa: E402

CANON = get_canonical_vertices().astype(np.float64)
O24 = geo.O24_rotations()

ENTRIES = [
    # (stem, label, category, input_field, original_metric_name)
    ("A1_gdrnet",                  "A1 GDR-Net",       "A", "pred_verts", "ADD-S"),
    ("A2_sc6d",                    "A2 SC6D",          "A", "pred_verts", "ADD-S"),
    ("A3_hccepose",                "A3 HccePose",      "A", "pred_verts", "ADD-S"),
    ("B1_rede",                    "B1 REDE",          "B", "kp3d_raw",   "MPJPE"),
    ("B2_uni6d",                   "B2 Uni6D",         "B", "kp3d_raw",   "MPJPE"),
    ("B3_ffb6d",                   "B3 FFB6D",         "B", "kp3d_raw",   "MPJPE"),
    ("C1_megapose_official",       "C1 MegaPose",      "C", "pred_verts", "ADD-S"),
    ("C2_gigapose_official",       "C2 GigaPose",      "C", "pred_verts", "ADD-S"),
    ("C3_foundationpose_official", "C3 FoundationPose","C", "pred_verts", "ADD-S"),
]


def sym_aware_mpjpe(model_verts: np.ndarray, gt_R: np.ndarray,
                    gt_t: np.ndarray, gt_s: float) -> float:
    """min over G in O_24 of mean per-vertex L2 distance."""
    best = np.inf
    for G in O24:
        gt_sym = gt_s * (CANON @ G.T) @ gt_R.T + gt_t
        m = float(np.linalg.norm(model_verts - gt_sym, axis=1).mean())
        if m < best:
            best = m
    return best


def compute_one(stem: str, field: str) -> dict:
    rec = json.load(open(PER_MODEL / f"{stem}_dump.json"))["records"]
    mpjpes: list[float] = []
    for r in rec:
        gt = np.asarray(r["gt_verts"], dtype=np.float64)
        mv = np.asarray(r[field],     dtype=np.float64)
        if gt.shape != (14, 3) or mv.shape != (14, 3):
            continue
        R_gt, t_gt, s_gt = geo.umeyama_np(CANON, gt)
        mpjpes.append(sym_aware_mpjpe(mv, R_gt, t_gt, float(s_gt)))
    arr = np.asarray(mpjpes)
    return {"n": int(len(arr)), "mean": float(arr.mean()), "std": float(arr.std())}


def main() -> None:
    print("\nStep 1 — Unified MPJPE computation (sym-aware over O_24).")
    print("All 9 models use the SAME formula; only input vertex source differs.\n")
    print(f"  {'Model':<19}  {'Cat':<3}  {'Input field':<11}  {'Original':<7}  "
          f"{'n':>6}  {'MPJPE (mean ± std, 4dp)':>30}")
    rows = []
    for stem, label, cat, field, orig in ENTRIES:
        info = compute_one(stem, field)
        rows.append({"stem": stem, "label": label, "cat": cat,
                     "field": field, "orig_name": orig,
                     "n": info["n"], "mean": info["mean"], "std": info["std"]})
        pm = f"{info['mean']:.4f} ± {info['std']:.4f}"
        print(f"  {label:<19}  {cat:<3}  {field:<11}  {orig:<7}  {info['n']:>6}  {pm:>30}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "label", "category", "input_field", "original_metric_name",
                    "n", "mpjpe_mean", "mpjpe_std"])
        for r in rows:
            w.writerow([r["stem"], r["label"], r["cat"], r["field"], r["orig_name"],
                        r["n"], f"{r['mean']:.4f}", f"{r['std']:.4f}"])
    print(f"\nsaved: {OUT_CSV.relative_to(ROOT)}")

    # Sanity: compare with prior corrected_addS_summary.csv if present
    prev = ROOT / "results" / "5.4Z_depth" / "corrected_addS_summary.csv"
    if prev.exists():
        print("\n--- sanity vs corrected_addS_summary.csv (raw_sym field) ---")
        prev_map = {}
        with open(prev) as f:
            for r in csv.DictReader(f):
                prev_map[r["stem"]] = (float(r["raw_sym_mean"]), float(r["raw_sym_std"]))
        print(f"  {'Model':<19}  {'unified_mpjpe (this)':>26}  "
              f"{'prior raw_sym':>26}  {'Δ mean':>10}")
        for r in rows:
            pm_now  = f"{r['mean']:.4f} ± {r['std']:.4f}"
            pmean, pstd = prev_map.get(r["stem"], (float('nan'), float('nan')))
            pm_old  = f"{pmean:.4f} ± {pstd:.4f}"
            d = r["mean"] - pmean
            print(f"  {r['label']:<19}  {pm_now:>26}  {pm_old:>26}  {d:>+10.6f}")


if __name__ == "__main__":
    main()
