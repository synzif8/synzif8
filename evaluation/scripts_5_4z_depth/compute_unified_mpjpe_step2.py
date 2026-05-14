"""Step 2 — Unified MPJPE table with both raw and Z-offset-corrected values.

For all 9 baselines (A/B/C), compute:
  raw  MPJPE  = sym-aware mean per-vertex L2 (no correction)
  MPJPE'      = sym-aware mean per-vertex L2 AFTER per-instance Z offset removed
  reduction%  = (raw_mean - corr_mean) / raw_mean * 100

All values reported as mean ± std (4 decimal places).

Outputs:
  - stdout : per-model table + LaTeX block
  - results/5.4Z_depth/unified_mpjpe_step2_summary.csv
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("<PROJECT_ROOT>/evaluation")
PER_MODEL = ROOT / "results" / "per_model"
OUT_CSV = ROOT / "results" / "5.4Z_depth" / "unified_mpjpe_step2_summary.csv"

sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, "<PROJECT_ROOT>")
from edge_utils import get_canonical_vertices  # type: ignore  # noqa: E402
from lineup.common import geometry as geo       # type: ignore  # noqa: E402

CANON = get_canonical_vertices().astype(np.float64)
O24 = geo.O24_rotations()

ENTRIES = [
    ("A1_gdrnet",                  "A1 GDR-Net",       "A", "pred_verts"),
    ("A2_sc6d",                    "A2 SC6D",          "A", "pred_verts"),
    ("A3_hccepose",                "A3 HccePose",      "A", "pred_verts"),
    ("B1_rede",                    "B1 REDE",          "B", "kp3d_raw"),
    ("B2_uni6d",                   "B2 Uni6D",         "B", "kp3d_raw"),
    ("B3_ffb6d",                   "B3 FFB6D",         "B", "kp3d_raw"),
    ("C1_megapose_official",       "C1 MegaPose",      "C", "pred_verts"),
    ("C2_gigapose_official",       "C2 GigaPose",      "C", "pred_verts"),
    ("C3_foundationpose_official", "C3 FoundationPose","C", "pred_verts"),
]


def sym_aware_mpjpe(verts: np.ndarray, gt_R: np.ndarray, gt_t: np.ndarray, gt_s: float) -> float:
    best = np.inf
    for G in O24:
        gt_sym = gt_s * (CANON @ G.T) @ gt_R.T + gt_t
        m = float(np.linalg.norm(verts - gt_sym, axis=1).mean())
        if m < best:
            best = m
    return best


def compute_one(stem: str, field: str) -> dict:
    rec = json.load(open(PER_MODEL / f"{stem}_dump.json"))["records"]
    raw_pi: list[float] = []
    cor_pi: list[float] = []
    for r in rec:
        gt = np.asarray(r["gt_verts"], dtype=np.float64)
        mv = np.asarray(r[field],     dtype=np.float64)
        if gt.shape != (14, 3) or mv.shape != (14, 3):
            continue
        R_gt, t_gt, s_gt = geo.umeyama_np(CANON, gt)
        raw_pi.append(sym_aware_mpjpe(mv, R_gt, t_gt, float(s_gt)))
        # per-instance Z offset removal
        dz_mean = float((mv[:, 2] - gt[:, 2]).mean())
        mv_c = mv.copy(); mv_c[:, 2] -= dz_mean
        cor_pi.append(sym_aware_mpjpe(mv_c, R_gt, t_gt, float(s_gt)))
    raw = np.asarray(raw_pi)
    cor = np.asarray(cor_pi)
    return {
        "n": int(len(raw)),
        "raw_mean":  float(raw.mean()),  "raw_std":  float(raw.std()),
        "corr_mean": float(cor.mean()),  "corr_std": float(cor.std()),
        "red_pct": float(100.0 * (raw.mean() - cor.mean()) / raw.mean()),
    }


def fmt_pm(m: float, s: float) -> str:
    return f"{m:.4f} ± {s:.4f}"


def main() -> None:
    print("\nStep 2 — Unified MPJPE table (raw + Z-offset corrected) — mean ± std, 4dp.\n")
    raw_hdr  = "MPJPE (raw)"
    corr_hdr = "MPJPE' (corrected)"
    print(f"  {'Model':<19}  {'Cat':<3}  {'n':>6}  "
          f"{raw_hdr:>26}  {corr_hdr:>26}  {'red%':>7}")
    rows = []
    for stem, label, cat, field in ENTRIES:
        info = compute_one(stem, field)
        rows.append({"stem": stem, "label": label, "cat": cat, "field": field, **info})
        print(f"  {label:<19}  {cat:<3}  {info['n']:>6}  "
              f"{fmt_pm(info['raw_mean'], info['raw_std']):>26}  "
              f"{fmt_pm(info['corr_mean'], info['corr_std']):>26}  "
              f"{info['red_pct']:>6.2f}%")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["stem", "label", "category", "input_field", "n",
                    "mpjpe_raw_mean", "mpjpe_raw_std",
                    "mpjpe_corr_mean", "mpjpe_corr_std",
                    "reduction_pct"])
        for r in rows:
            w.writerow([r["stem"], r["label"], r["cat"], r["field"], r["n"],
                        f"{r['raw_mean']:.4f}",  f"{r['raw_std']:.4f}",
                        f"{r['corr_mean']:.4f}", f"{r['corr_std']:.4f}",
                        f"{r['red_pct']:.4f}"])
    print(f"\nsaved: {OUT_CSV.relative_to(ROOT)}")

    # ---------- LaTeX block ----------
    by = {r["stem"]: r for r in rows}
    A = ["A3_hccepose", "A2_sc6d", "A1_gdrnet"]
    B = ["B1_rede", "B3_ffb6d", "B2_uni6d"]
    C = ["C2_gigapose_official", "C3_foundationpose_official", "C1_megapose_official"]

    def best_of(stems, key):
        return stems[int(np.argmin([by[s][key] for s in stems]))]

    A_b_raw  = best_of(A, "raw_mean");  A_b_cor  = best_of(A, "corr_mean")
    B_b_raw  = best_of(B, "raw_mean");  B_b_cor  = best_of(B, "corr_mean")
    C_b_raw  = best_of(C, "raw_mean");  C_b_cor  = best_of(C, "corr_mean")

    def cell(stem, m_key, s_key, bold_stem):
        v = f"${by[stem][m_key]:.4f}\\,\\pm\\,{by[stem][s_key]:.4f}$"
        return rf"\textbf{{{v}}}" if stem == bold_stem else v

    L = []
    L.append(r"\begin{table}[!htbp]")
    L.append(r"\centering")
    L.append(r"\caption{Unified pose error on synthetic data, before and after per-instance $z$ offset correction (Eq.~\ref{eq:per_instance_z_offset}). All entries report MPJPE (sym-aware mean per-vertex L2 over the $O_{24}$ rotation group of the rhombic dodecahedron, $n=11{,}379$, mean $\pm$ std). The reduction from MPJPE to MPJPE$'$ quantifies the share of error concentrated in the undeterminable depth translation.}")
    L.append(r"\label{tab:z-depth_adaption}")
    L.append(r"\small")
    L.append(r"\setlength{\tabcolsep}{10pt}")
    L.append(r"\begin{tabular}{@{}lcc@{}}")
    L.append(r"\toprule")
    L.append(r"Model & MPJPE (px)$\downarrow$ & MPJPE$'$ (px)$\downarrow$ \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{3}{l}{\textit{(A) RGB regression}} \\")
    L.append(r"\midrule")
    L.append(rf"HccePose~\cite{{wang2025hccepose}} & {cell('A3_hccepose','raw_mean','raw_std',A_b_raw)} & {cell('A3_hccepose','corr_mean','corr_std',A_b_cor)} \\")
    L.append(rf"SC6D~\cite{{cai2022sc6d}}          & {cell('A2_sc6d','raw_mean','raw_std',A_b_raw)} & {cell('A2_sc6d','corr_mean','corr_std',A_b_cor)} \\")
    L.append(rf"GDR-Net~\cite{{wang2021gdr}}       & {cell('A1_gdrnet','raw_mean','raw_std',A_b_raw)} & {cell('A1_gdrnet','corr_mean','corr_std',A_b_cor)} \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{3}{l}{\textit{(B) 3D keypoint regression}} \\")
    L.append(r"\midrule")
    L.append(rf"REDE~\cite{{hua2021rede}}      & {cell('B1_rede','raw_mean','raw_std',B_b_raw)} & {cell('B1_rede','corr_mean','corr_std',B_b_cor)} \\")
    L.append(rf"FFB6D~\cite{{he2021ffb6d}}     & {cell('B3_ffb6d','raw_mean','raw_std',B_b_raw)} & {cell('B3_ffb6d','corr_mean','corr_std',B_b_cor)} \\")
    L.append(rf"Uni6D~\cite{{jiang2022uni6d}}  & {cell('B2_uni6d','raw_mean','raw_std',B_b_raw)} & {cell('B2_uni6d','corr_mean','corr_std',B_b_cor)} \\")
    L.append(r"\midrule")
    L.append(r"\multicolumn{3}{l}{\textit{(C) Foundation pose --- fine-tuned}} \\")
    L.append(r"\midrule")
    L.append(rf"GigaPose~\cite{{nguyen2024gigapose}}            & {cell('C2_gigapose_official','raw_mean','raw_std',C_b_raw)} & {cell('C2_gigapose_official','corr_mean','corr_std',C_b_cor)} \\")
    L.append(rf"FoundationPose~\cite{{wen2024foundationpose}}   & {cell('C3_foundationpose_official','raw_mean','raw_std',C_b_raw)} & {cell('C3_foundationpose_official','corr_mean','corr_std',C_b_cor)} \\")
    L.append(rf"MegaPose~\cite{{labbe2022megapose}}             & {cell('C1_megapose_official','raw_mean','raw_std',C_b_raw)} & {cell('C1_megapose_official','corr_mean','corr_std',C_b_cor)} \\")
    L.append(r"\bottomrule")
    L.append(r"\end{tabular}")
    L.append(r"\end{table}")

    print("\n--- LaTeX (paste-ready) ---\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
