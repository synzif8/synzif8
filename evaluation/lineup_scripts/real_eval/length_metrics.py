"""Compute edge / diffusivity / permeability metrics on the new 70-image
real-SEM length-annotation set (n=717).

GT  : 2D annotated edge length per rhombic (`length_px` from the new annotation
      JSONs).
Pred: model-estimated 3D edge length:
   - A / C  : pred_verts = s · BV @ R.T + t  →  mean of 24 canonical edges  (==  s · √3
              for these models since BV is rigid; but we follow the same code path
              as the existing `eval_with_maskrcnn_500.py` for consistency).
   - B (raw): kp3d_raw (14×3 raw keypoint output, no Procrustes) →  mean of 24
              canonical edges  ("raw 24" path, matching `edge_d_p_ranking_two_sets.md`).

Output table mirrors `edge_length_experiment/results/edge_d_p_ranking_two_sets.md`
(Set B — 407 instance, human-aligned) format.
"""

from __future__ import annotations

import json
import os
import sys
from glob import glob

import numpy as np

REPO = "<PROJECT_ROOT>/baselines"
sys.path.insert(0, REPO)

PRED_DIR = os.path.join(REPO, "runs/full/_length_eval/preds")
CONSENSUS_SUMMARY = os.path.join(REPO, "runs/full/_length_eval/viz_consensus/_summary.json")
VISIBLE_MESH_SUMMARY = os.path.join(REPO, "runs/full/_length_eval/viz_visible_mesh/_summary.json")
OUT_MD = os.path.join(REPO, "runs/full/_length_eval/EDGE_D_P_RANKING.md")
OUT_JSON = os.path.join(REPO, "runs/full/_length_eval/EDGE_D_P_RANKING.json")

# Canonical 14×3 + 24 edges (same convention as edge_length_experiment).
BASE_VERTICES = np.array([
    [0.0,  0.0, -2.0], [0.0,  2.0,  0.0], [-2.0, 0.0,  0.0], [2.0,  0.0,  0.0],
    [0.0, -2.0,  0.0], [1.0,  1.0,  1.0], [-1.0, 1.0,  1.0], [1.0, -1.0,  1.0],
    [0.0,  0.0,  2.0], [-1.0,-1.0,  1.0], [1.0,  1.0, -1.0], [-1.0, 1.0, -1.0],
    [1.0, -1.0, -1.0], [-1.0,-1.0, -1.0],
], dtype=np.float64)

FACES = [
    [8, 5, 1, 6], [8, 6, 2, 9], [8, 9, 4, 7], [8, 7, 3, 5],
    [0, 10, 1, 11], [0, 11, 2, 13], [0, 13, 4, 12], [0, 12, 3, 10],
    [1, 5, 3, 10], [3, 7, 4, 12], [4, 9, 2, 13], [2, 6, 1, 11],
]


def get_edges() -> np.ndarray:
    e = set()
    for face in FACES:
        for i in range(len(face)):
            a, b = face[i], face[(i + 1) % len(face)]
            e.add(tuple(sorted((a, b))))
    out = sorted(e)
    assert len(out) == 24
    return np.array(out, dtype=np.int32)


EDGES = get_edges()                              # (24, 2)


def edge_lengths_3d(verts: np.ndarray) -> np.ndarray:
    """(N, 14, 3) -> (N, 24)."""
    a = verts[:, EDGES[:, 0], :]
    b = verts[:, EDGES[:, 1], :]
    return np.linalg.norm(a - b, axis=-1)


def edge_lengths_2d(verts_2d: np.ndarray) -> np.ndarray:
    """(N, 14, 2) -> (N, 24)."""
    a = verts_2d[:, EDGES[:, 0], :]
    b = verts_2d[:, EDGES[:, 1], :]
    return np.linalg.norm(a - b, axis=-1)


def metrics_dict(lp: np.ndarray, lg: np.ndarray) -> dict:
    delta = lp - lg
    rel = delta / lg
    ratio2 = (lp / lg) ** 2 - 1.0
    abs_delta = np.abs(delta)
    abs_rel_pct = np.abs(rel) * 100.0
    abs_ratio2_pct = np.abs(ratio2) * 100.0
    return dict(
        n=int(len(lp)),
        edge_mae=float(np.mean(abs_delta)),
        edge_mae_std=float(np.std(abs_delta)),
        edge_rmse=float(np.sqrt(np.mean(delta ** 2))),
        edge_rmse_std=float(np.std(delta)),
        edge_mape_pct=float(np.mean(abs_rel_pct)),
        edge_mape_pct_std=float(np.std(abs_rel_pct)),
        edge_rel_rmse_pct=float(np.sqrt(np.mean(rel ** 2)) * 100.0),
        edge_rel_rmse_pct_std=float(np.std(rel) * 100.0),
        d_mae_pct=float(np.mean(abs_ratio2_pct)),
        d_mae_pct_std=float(np.std(abs_ratio2_pct)),
        d_rmse_pct=float(np.sqrt(np.mean(ratio2 ** 2)) * 100.0),
        d_rmse_pct_std=float(np.std(ratio2) * 100.0),
    )


# Display label / category for each model.
DISPLAY = {
    "A1_gdrnet":                  ("A1 GDR-Net",          "A"),
    "A2_sc6d":                    ("A2 SC6D",             "A"),
    "A3_hccepose":                ("A3 HccePose",         "A"),
    "B1_rede":                    ("B1 REDE (raw 24)",    "B"),
    "B2_uni6d":                   ("B2 Uni6D (raw 24)",   "B"),
    "B3_ffb6d":                   ("B3 FFB6D (raw 24)",   "B"),
    "C1_megapose_official":       ("C1 MegaPose",         "C"),
    "C2_gigapose_official":       ("C2 GigaPose",         "C"),
    "C3_foundationpose_official": ("C3 FoundationPose",   "C"),
}


def collect() -> dict:
    """For each model, accumulate (l_pred, l_gt) across all rhombics in all NPZs."""
    files = sorted(glob(os.path.join(PRED_DIR, "*.npz")))
    n_objects_total = 0

    per_model_lp: dict[str, list[float]] = {n: [] for n in DISPLAY}
    per_model_lg: dict[str, list[float]] = {n: [] for n in DISPLAY}

    for fp in files:
        d = np.load(fp, allow_pickle=False)
        names = [str(n) for n in d["model_names"]]
        verts_cam_full = None  # (N, M, 14, 3) when needed
        # Reconstruct 3D camera-frame vertices for A/C: s · BV @ R.T + t.
        N, M = d["R"].shape[:2]
        canonical = BASE_VERTICES.astype(np.float32)             # (14, 3)
        verts_cam = (
            d["s"][..., None, None] * np.einsum("nmij,vj->nmvi", d["R"], canonical)
            + d["t"][:, :, None, :]
        )                                                        # (N, M, 14, 3)
        kp3d = d["kp3d_raw"]                                      # (N, M, 14, 3) NaN-padded
        gt_len = d["length_px"]                                   # (N,)
        n_objects_total += N

        for mi, name in enumerate(names):
            if name not in per_model_lp:
                continue
            cat = DISPLAY[name][1]
            if cat == "B":
                # Use raw kp3d (14×3) — fall back to verts_cam if NaN (shouldn't happen for B).
                kp = kp3d[:, mi]                                  # (N, 14, 3)
                if np.isnan(kp).any():
                    kp = verts_cam[:, mi]
            else:
                kp = verts_cam[:, mi]                             # (N, 14, 3)
            edge_lens = edge_lengths_3d(kp)                       # (N, 24)
            mean_edges = edge_lens.mean(axis=1)                   # (N,)
            per_model_lp[name].extend(mean_edges.tolist())
            per_model_lg[name].extend(gt_len.tolist())

    summary = {"n_objects_total": n_objects_total, "per_model": {}}
    for name in DISPLAY:
        lp = np.array(per_model_lp[name], dtype=np.float64)
        lg = np.array(per_model_lg[name], dtype=np.float64)
        if lp.size == 0:
            continue
        m = metrics_dict(lp, lg)
        summary["per_model"][name] = m
    return summary


def collect_consensus_2d() -> dict:
    """Aggregate 2D consensus-edge predictions from `viz_consensus/_summary.json`.

    For each rhombic, the consensus pipeline (A+C voting) selects a single
    canonical edge `(id_a, id_b)` shared by all 9 models; per-model `pred_length_px`
    is the 2D distance between the matched vertices after O48 best-σ selection.

    Returns: {
        "full": per-model metrics on the full 717 set,
        "tier_canonical": per-model metrics restricted to rhombics where the
                          consensus picked a *real* canonical edge (n=542),
        "tier_canonical_inliers": canonical AND inliers≥5/6 (n=465),
        "n_full", "n_tier_canonical", "n_tier_canonical_inliers"
    }
    """
    with open(CONSENSUS_SUMMARY) as f:
        s = json.load(f)

    full_lp = {n: [] for n in DISPLAY}
    full_lg = {n: [] for n in DISPLAY}
    tier_can_lp = {n: [] for n in DISPLAY}
    tier_can_lg = {n: [] for n in DISPLAY}
    tier_both_lp = {n: [] for n in DISPLAY}
    tier_both_lg = {n: [] for n in DISPLAY}

    n_full = 0; n_can = 0; n_both = 0
    for stem, img_data in s["images"].items():
        for obj in img_data["per_object"]:
            gl = float(obj["gt_length_px"])
            is_can = bool(obj["consensus_edge_canonical"])
            is_inl = int(obj["consensus_inlier_count"]) >= 5
            n_full += 1
            if is_can: n_can += 1
            if is_can and is_inl: n_both += 1
            for m in obj["models"]:
                name = m["name"]
                if name not in full_lp:
                    continue
                lp = float(m["pred_length_px"])
                full_lp[name].append(lp)
                full_lg[name].append(gl)
                if is_can:
                    tier_can_lp[name].append(lp)
                    tier_can_lg[name].append(gl)
                if is_can and is_inl:
                    tier_both_lp[name].append(lp)
                    tier_both_lg[name].append(gl)

    out = {
        "n_full": n_full,
        "n_tier_canonical": n_can,
        "n_tier_canonical_inliers": n_both,
        "full": {}, "tier_canonical": {}, "tier_canonical_inliers": {},
    }
    for name in DISPLAY:
        if full_lp[name]:
            out["full"][name] = metrics_dict(np.array(full_lp[name]), np.array(full_lg[name]))
        if tier_can_lp[name]:
            out["tier_canonical"][name] = metrics_dict(np.array(tier_can_lp[name]),
                                                        np.array(tier_can_lg[name]))
        if tier_both_lp[name]:
            out["tier_canonical_inliers"][name] = metrics_dict(
                np.array(tier_both_lp[name]), np.array(tier_both_lg[name]))
    return out


def collect_longest_2d_no_visibility() -> dict:
    """Set 4: max 2D-projected edge among all 24 canonical edges (no visibility filter).

    - A / C : pred_verts (3D) = s · BV @ R.T + t  →  2D = drop z  →  max of 24 edges.
    - B     : kp3d_raw (3D, raw keypoints — no Procrustes)  →  drop z  →  max of 24 edges.

    For centrally-symmetric prediction (A/C, since BV is centrally symmetric and
    similarity transforms preserve it), antipodal edges share identical 2D length, so
    Set 4 max == Set 3 max numerically. For B (free keypoints), Set 4 may differ.
    """
    files = sorted(glob(os.path.join(PRED_DIR, "*.npz")))

    per_model_lp: dict[str, list[float]] = {n: [] for n in DISPLAY}
    per_model_lg: dict[str, list[float]] = {n: [] for n in DISPLAY}

    n_full = 0
    for fp in files:
        d = np.load(fp, allow_pickle=False)
        names = [str(n) for n in d["model_names"]]
        N, M = d["R"].shape[:2]
        canonical = BASE_VERTICES.astype(np.float32)
        # A/C 3D verts (Procrustes-decoded)
        verts_cam = (
            d["s"][..., None, None] * np.einsum("nmij,vj->nmvi", d["R"], canonical)
            + d["t"][:, :, None, :]
        )                                                        # (N, M, 14, 3)
        kp3d = d["kp3d_raw"]                                      # (N, M, 14, 3)
        gt_len = d["length_px"]                                   # (N,)
        n_full += N

        for mi, name in enumerate(names):
            if name not in per_model_lp:
                continue
            cat = DISPLAY[name][1]
            if cat == "B":
                kp = kp3d[:, mi]                                  # (N, 14, 3)
                if np.isnan(kp).any():
                    kp = verts_cam[:, mi]
            else:
                kp = verts_cam[:, mi]
            verts_2d = kp[..., :2]                                # (N, 14, 2)
            lens_2d = edge_lengths_2d(verts_2d)                   # (N, 24)
            max_2d = lens_2d.max(axis=1)                          # (N,)
            per_model_lp[name].extend(max_2d.tolist())
            per_model_lg[name].extend(gt_len.tolist())

    out = {"n_full": n_full, "full": {}}
    for name in DISPLAY:
        if per_model_lp[name]:
            out["full"][name] = metrics_dict(np.array(per_model_lp[name]),
                                              np.array(per_model_lg[name]))
    return out


def collect_visible_longest_2d() -> dict:
    """Aggregate Set 3: Pred = longest 2D-projected visible edge per (object, model).

    Visible = edge with at least one front-facing incident face (face-centroid_z > -ε
    after rotation). Pred is the maximum 2D Euclidean length among such edges.

    Returns: {
        "n_full": 717,
        "n_tier_clean": # objects where every model has F=6 (no tilted-view fallback),
        "full":        per-model metrics on full 717 set,
        "tier_clean":  per-model metrics on the F=6 (canonical 6/12 split) subset,
    }
    """
    with open(VISIBLE_MESH_SUMMARY) as f:
        s = json.load(f)

    full_lp = {n: [] for n in DISPLAY}
    full_lg = {n: [] for n in DISPLAY}
    tier_lp = {n: [] for n in DISPLAY}
    tier_lg = {n: [] for n in DISPLAY}

    n_full = 0
    n_tier_clean = 0
    for stem, img_data in s["images"].items():
        for obj in img_data["per_object"]:
            gl = float(obj["gt_length_px"])
            n_full += 1
            # Tier-clean: ALL 9 models report F=6 (no tilt fallback) for this object.
            all_f6 = all(int(m["n_front_faces"]) == 6 for m in obj["models"])
            if all_f6:
                n_tier_clean += 1
            for m in obj["models"]:
                name = m["name"]
                if name not in full_lp:
                    continue
                lp = float(m["longest_visible_edge_2d_px"])
                full_lp[name].append(lp)
                full_lg[name].append(gl)
                if all_f6:
                    tier_lp[name].append(lp)
                    tier_lg[name].append(gl)

    out = {"n_full": n_full, "n_tier_clean": n_tier_clean,
           "full": {}, "tier_clean": {}}
    for name in DISPLAY:
        if full_lp[name]:
            out["full"][name] = metrics_dict(np.array(full_lp[name]), np.array(full_lg[name]))
        if tier_lp[name]:
            out["tier_clean"][name] = metrics_dict(np.array(tier_lp[name]), np.array(tier_lg[name]))
    return out


def _format_metric_row(rank: int, display: str, cat: str, m: dict) -> str:
    return (
        f"| {rank} | {display} | {cat} | {m['n']} | "
        f"{m['edge_mae']:.4f} ± {m['edge_mae_std']:.4f} | "
        f"{m['edge_rmse']:.4f} ± {m['edge_rmse_std']:.4f} | "
        f"{m['edge_mape_pct']:.4f} ± {m['edge_mape_pct_std']:.4f} | "
        f"{m['edge_rel_rmse_pct']:.4f} ± {m['edge_rel_rmse_pct_std']:.4f} | "
        f"{m['d_mae_pct']:.4f} ± {m['d_mae_pct_std']:.4f} | "
        f"{m['d_rmse_pct']:.4f} ± {m['d_rmse_pct_std']:.4f} |"
    )


def _ranked_rows(per_model: dict) -> list[tuple[int, str, str, dict]]:
    rows = []
    for name, (display, cat) in DISPLAY.items():
        m = per_model.get(name)
        if m is None:
            continue
        rows.append((cat, m["edge_rel_rmse_pct"], display, name, m))
    rows.sort(key=lambda r: r[1])
    return [(i + 1, r[2], r[0], r[4]) for i, r in enumerate(rows)]


def render_md(summary: dict, consensus: dict, visible: dict, no_vis: dict) -> str:
    L = []
    L.append("# Edge / Diffusivity / Permeability — Real-SEM 70-image (n=717)\n")
    L.append("- Date: derived from `runs/full/_length_eval/preds/`, "
             "`runs/full/_length_eval/viz_consensus/_summary.json`, "
             "`runs/full/_length_eval/viz_visible_mesh/_summary.json`")
    L.append("- **GT (모든 표 공통)**: 2D annotated edge length (`length_px` from the user's "
             "vertex-annotator length tool, 1 click-pair per rhombic, n=717).")
    L.append("- **Pred (Set 1 — 3D 24-edge mean)**:")
    L.append("  - A / C: `pred_verts = s · BV @ R.T + t`, 24 canonical edge 평균 (= s · √3 ).")
    L.append("  - B    : `kp3d_pred` (raw, no Procrustes) 14×3, 24 canonical edge 평균 "
             "(\"raw 24\" 경로, `edge_d_p_ranking_two_sets.md`와 동일 정의).")
    L.append("- **Pred (Set 2 — 2D matched-edge)**: A+C 6-model voting consensus 로 GT 2D 클릭 "
             "(p1, p2)에 대응하는 canonical edge `(id_a, id_b)`를 한 번 결정 → 모든 9 모델에 동일 "
             "ID 적용 → 각 모델의 14×2 출력에서 O48 best σ (z-flip 포함) + visibility 필터 적용해 "
             "matched 2D edge endpoint를 잡고 그 2D 거리를 사용.")
    L.append("  - **공정 비교**: 모든 모델이 같은 canonical edge를 평가받음.")
    L.append("  - **차원 일치**: GT (2D px) ↔ Pred (2D px) — Set 1과 달리 단위가 같음.")
    L.append("- **Pred (Set 3 — longest visible 2D edge)**: 모델의 14×3 예측에서 face 단위 visibility "
             "판정 (`(R @ face_centroid)_z > -ε`, ε=0.05) → 최소 한 face가 front인 edge만 \"visible\" "
             "→ 그 visible edge들을 2D에 직교 투영(z 제거) → **최장 2D 길이**가 모델의 edge length.")
    L.append("  - 차원 일치 (2D px), 매칭 ID 불필요, consensus 의존 없음 — 가장 *method-independent*.")
    L.append("  - 단, GT(특정 1개 edge)와 Pred(visible 중 최대값)의 정의 비대칭 → 약한 *upward bias* 가능.")
    L.append("- **Pred (Set 4 — longest 2D edge, no visibility)**: 24개 canonical edge 모두 2D 투영 "
             "후 단순 max — visibility 무시.")
    L.append("  - A/C는 강체 + 중심대칭이라 Set 3과 수치적으로 동일 (back-edge가 front-edge와 동일한 "
             "2D 길이를 가짐).")
    L.append("  - B는 raw kp3d가 자유 14×3이므로 중심대칭이 깨질 수 있어 Set 3과 차이가 날 수 있음.")
    L.append("- Sort: ascending **Edge rel RMSE [%]** (lower = better).")
    L.append("- ± 표기: per-instance scalar의 표준편차 (instance-level 변동, training run variance 아님).")
    L.append("  - MAE / MAPE / D · P MAE: σ(|·|)")
    L.append("  - RMSE / rel-RMSE / D · P RMSE: σ(signed error)\n")

    L.append("## Notation")
    L.append(r"- $l_{pred,i}$: i-th instance predicted edge length [px]")
    L.append(r"- $l_{gt,i}$  : i-th instance GT 2D edge length [px]  (one user-clicked edge)")
    L.append(r"- $\Delta l_i = l_{pred,i} - l_{gt,i}$\n")

    L.append("## 사용 수식")
    L.append(r"$\text{MAE}_{\text{edge}} = \frac{1}{N}\sum |l_{pred,i} - l_{gt,i}| \quad [\text{px}]$  "
             r"$\quad\text{RMSE}_{\text{edge,abs}} = \sqrt{\frac{1}{N}\sum (l_{pred,i} - l_{gt,i})^2} \quad [\text{px}]$")
    L.append(r"$\text{MAPE}_{\text{edge}} = \frac{1}{N}\sum |\frac{\Delta l_i}{l_{gt,i}}| \times 100 \quad [\%]$  "
             r"$\quad\text{RMSE}_{\text{edge,rel}} = \sqrt{\frac{1}{N}\sum (\frac{\Delta l_i}{l_{gt,i}})^2} \times 100 \quad [\%]$")
    L.append(r"$\text{MAE}_D = \frac{1}{N}\sum |(\frac{l_{pred,i}}{l_{gt,i}})^2 - 1| \times 100$  "
             r"$\quad\text{RMSE}_D = \sqrt{\frac{1}{N}\sum ((\frac{l_{pred,i}}{l_{gt,i}})^2 - 1)^2} \times 100 \quad [\%]$")
    L.append("")

    # ---- Set 1 — 3D 24-edge mean ----
    L.append(f"## Set 1 — 3D 24-edge mean (n={summary['n_objects_total']})\n")
    L.append("Pred = 모델이 추정한 3D rhombic edge 평균 길이. GT (2D)와 차원이 다르므로 *bias term*이 "
             "포함됨에 유의 (3D Euclidean ≥ 2D projection). 같은 baseline을 모든 모델에 적용했으므로 "
             "상대 비교는 공정.\n")
    L.append("| Rank | Model | Cat | n | Edge MAE [px] | Edge RMSE [px] | "
             "Edge MAPE [%] | Edge rel RMSE [%] | D / P MAE [%] | D / P RMSE [%] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for rank, display, cat, m in _ranked_rows(summary["per_model"]):
        L.append(_format_metric_row(rank, display, cat, m))
    L.append("")

    # ---- Set 2 — 2D matched-edge (consensus) ----
    L.append(f"## Set 2 — 2D matched-edge, A+C consensus IDs, O48 best σ (n={consensus['n_full']})\n")
    L.append("Pred = consensus가 지정한 canonical edge `(id_a, id_b)`에 대한 모델별 2D edge length. "
             "GT와 같은 2D 픽셀 단위 → bias 없음. 단, consensus 자체가 24.4% 객체에서 비-canonical "
             "edge를 잡았으므로 (face-diagonal 등) 신뢰 가능한 부분집합 게이팅 권장.\n")
    L.append("| Rank | Model | Cat | n | Edge MAE [px] | Edge RMSE [px] | "
             "Edge MAPE [%] | Edge rel RMSE [%] | D / P MAE [%] | D / P RMSE [%] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for rank, display, cat, m in _ranked_rows(consensus["full"]):
        L.append(_format_metric_row(rank, display, cat, m))
    L.append("")

    # ---- Set 2-Tier1 — consensus_edge_canonical ----
    L.append(f"### Set 2 (Tier-1 — consensus가 진짜 canonical edge 선택) "
             f"(n={consensus['n_tier_canonical']})\n")
    L.append("`consensus_edge_canonical=True` 만 포함. 매칭 자체가 의미 있는 부분집합.\n")
    L.append("| Rank | Model | Cat | n | Edge MAE [px] | Edge RMSE [px] | "
             "Edge MAPE [%] | Edge rel RMSE [%] | D / P MAE [%] | D / P RMSE [%] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for rank, display, cat, m in _ranked_rows(consensus["tier_canonical"]):
        L.append(_format_metric_row(rank, display, cat, m))
    L.append("")

    # ---- Set 2-Tier2 — canonical AND inliers≥5 ----
    L.append(f"### Set 2 (Tier-2 — canonical + A+C inlier ≥ 5 / 6) "
             f"(n={consensus['n_tier_canonical_inliers']})\n")
    L.append("진짜 edge 매칭 + A+C 6모델 중 5개 이상 합의한 강한 신뢰 부분집합.\n")
    L.append("| Rank | Model | Cat | n | Edge MAE [px] | Edge RMSE [px] | "
             "Edge MAPE [%] | Edge rel RMSE [%] | D / P MAE [%] | D / P RMSE [%] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for rank, display, cat, m in _ranked_rows(consensus["tier_canonical_inliers"]):
        L.append(_format_metric_row(rank, display, cat, m))
    L.append("")

    # ---- Set 3 — longest visible 2D edge ----
    L.append(f"## Set 3 — longest visible 2D edge (n={visible['n_full']})\n")
    L.append("Pred = 모델이 예측한 3D rhombic에서 face-기반 visibility 추출 후 visible edge 중 2D 투영 "
             "최장값. 매칭 ID도 consensus도 필요 없음 — 모델 자체 출력만 사용. 차원 일치 (2D vs 2D).\n")
    L.append("| Rank | Model | Cat | n | Edge MAE [px] | Edge RMSE [px] | "
             "Edge MAPE [%] | Edge rel RMSE [%] | D / P MAE [%] | D / P RMSE [%] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for rank, display, cat, m in _ranked_rows(visible["full"]):
        L.append(_format_metric_row(rank, display, cat, m))
    L.append("")

    # ---- Set 3 Tier — clean F=6 view across all models ----
    L.append(f"### Set 3 (Tier — 9 모델 모두 F=6 face 인 \"clean view\" 객체) "
             f"(n={visible['n_tier_clean']})\n")
    L.append("객체가 카메라에 정면 가까이 있어 모든 모델이 6/12 face split (정확한 절반 가시성)을 "
             "내놓은 부분집합. tilt 케이스 제외.\n")
    L.append("| Rank | Model | Cat | n | Edge MAE [px] | Edge RMSE [px] | "
             "Edge MAPE [%] | Edge rel RMSE [%] | D / P MAE [%] | D / P RMSE [%] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for rank, display, cat, m in _ranked_rows(visible["tier_clean"]):
        L.append(_format_metric_row(rank, display, cat, m))
    L.append("")

    # ---- Set 4 — longest 2D edge, no visibility ----
    L.append(f"## Set 4 — longest 2D edge, no visibility filter (n={no_vis['n_full']})\n")
    L.append("Pred = 24 canonical edge 모두 2D 투영 후 max. Visibility 무시 → 가장 단순한 정의.\n")
    L.append("| Rank | Model | Cat | n | Edge MAE [px] | Edge RMSE [px] | "
             "Edge MAPE [%] | Edge rel RMSE [%] | D / P MAE [%] | D / P RMSE [%] |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for rank, display, cat, m in _ranked_rows(no_vis["full"]):
        L.append(_format_metric_row(rank, display, cat, m))
    L.append("")

    # ---- Set 1 vs Set 2 vs Set 3 vs Set 4 comparison ----
    L.append("## Set 1 vs Set 2 vs Set 3 vs Set 4 (Full) — Pred 정의 비교\n")
    L.append("같은 717 인스턴스에서 Pred 정의만 다를 때의 metric 변화. 모든 metric은 \"같은 GT\" "
             "(2D annotated length) 대비.\n")
    L.append("| Model | Cat | "
             "S1 rel-RMSE | S2 rel-RMSE | S3 rel-RMSE | S4 rel-RMSE | "
             "S1 D-RMSE | S2 D-RMSE | S3 D-RMSE | S4 D-RMSE |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for name, (display, cat) in DISPLAY.items():
        m1 = summary["per_model"].get(name)
        m2 = consensus["full"].get(name)
        m3 = visible["full"].get(name)
        m4 = no_vis["full"].get(name)
        if any(m is None for m in (m1, m2, m3, m4)):
            continue
        L.append(f"| {display} | {cat} | "
                 f"{m1['edge_rel_rmse_pct']:.2f} | {m2['edge_rel_rmse_pct']:.2f} | "
                 f"{m3['edge_rel_rmse_pct']:.2f} | {m4['edge_rel_rmse_pct']:.2f} | "
                 f"{m1['d_rmse_pct']:.2f} | {m2['d_rmse_pct']:.2f} | "
                 f"{m3['d_rmse_pct']:.2f} | {m4['d_rmse_pct']:.2f} |")
    L.append("")

    # ---- Set 3 vs Set 4 — visibility 효과 ----
    L.append("## Set 3 vs Set 4 — visibility 필터 효과 검증\n")
    L.append("A/C는 중심대칭이라 antipodal edge 짝이 같은 2D 길이 → 수치 동일 예상. "
             "B는 raw kp3d가 자유 14×3이라 차이가 날 수 있음.\n")
    L.append("| Model | Cat | S3 rel-RMSE | S4 rel-RMSE | Δ (S4 − S3) | "
             "S3 D-RMSE | S4 D-RMSE | Δ |")
    L.append("|---|---|---|---|---|---|---|---|")
    for name, (display, cat) in DISPLAY.items():
        m3 = visible["full"].get(name)
        m4 = no_vis["full"].get(name)
        if m3 is None or m4 is None:
            continue
        d_e = m4["edge_rel_rmse_pct"] - m3["edge_rel_rmse_pct"]
        d_d = m4["d_rmse_pct"] - m3["d_rmse_pct"]
        L.append(f"| {display} | {cat} | "
                 f"{m3['edge_rel_rmse_pct']:.4f} | {m4['edge_rel_rmse_pct']:.4f} | "
                 f"{d_e:+.4f} | "
                 f"{m3['d_rmse_pct']:.4f} | {m4['d_rmse_pct']:.4f} | "
                 f"{d_d:+.4f} |")
    L.append("")

    return "\n".join(L)


def main():
    summary = collect()
    consensus = collect_consensus_2d()
    visible = collect_visible_longest_2d()
    no_vis = collect_longest_2d_no_visibility()
    md = render_md(summary, consensus, visible, no_vis)
    with open(OUT_MD, "w") as f:
        f.write(md)
    with open(OUT_JSON, "w") as f:
        json.dump({"set1_3d_mean": summary,
                   "set2_2d_matched": consensus,
                   "set3_longest_visible_2d": visible,
                   "set4_longest_2d_no_visibility": no_vis}, f, indent=2)
    print(f"wrote {OUT_MD}")
    print(f"wrote {OUT_JSON}")
    print(f"\n--- table ---\n")
    print(md)


if __name__ == "__main__":
    main()
