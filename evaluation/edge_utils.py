"""Edge length 실험 공통 유틸.

read-only: lineup/ 코드에 의존하지 않음 (FACES와 base_vertices 값을 hardcode).
"""

from __future__ import annotations

import numpy as np


# create_dataset_v6.py:23 의 base_vertices 그대로
_BASE_VERTICES = np.array([
    [0.0,  0.0, -2.0], [0.0,  2.0,  0.0], [-2.0, 0.0,  0.0], [2.0,  0.0,  0.0],   # 0-3
    [0.0, -2.0,  0.0], [1.0,  1.0,  1.0], [-1.0, 1.0,  1.0], [1.0, -1.0,  1.0],   # 4-7
    [0.0,  0.0,  2.0], [-1.0,-1.0,  1.0], [1.0,  1.0, -1.0], [-1.0, 1.0, -1.0],   # 8-11
    [1.0, -1.0, -1.0], [-1.0,-1.0, -1.0],                                          # 12-13
], dtype=np.float64)

# rasterize_nocs.py / viz_no_leak.py / create_dataset_v6.py 동일
_FACES = [
    [8, 5, 1, 6], [8, 6, 2, 9], [8, 9, 4, 7], [8, 7, 3, 5],
    [0, 10, 1, 11], [0, 11, 2, 13], [0, 13, 4, 12], [0, 12, 3, 10],
    [1, 5, 3, 10], [3, 7, 4, 12], [4, 9, 2, 13], [2, 6, 1, 11],
]


def get_canonical_vertices() -> np.ndarray:
    """(14, 3) canonical rhombic-dodecahedron vertices."""
    return _BASE_VERTICES.copy()


def get_edge_pairs() -> list[tuple[int, int]]:
    """24 unique edges derived from FACES (face의 4-cycle을 edge로 분해 → dedup).

    Euler 검증: V - E + F = 14 - 24 + 12 = 2 (genus 0 polyhedron).
    """
    edges: set[tuple[int, int]] = set()
    for face in _FACES:
        n = len(face)
        for i in range(n):
            a, b = face[i], face[(i + 1) % n]
            edges.add(tuple(sorted((a, b))))
    out = sorted(edges)
    assert len(out) == 24, f"expected 24 edges, got {len(out)}"
    return out


def compute_edge_lengths_3d(verts: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    """(N, 14, 3) → (N, 24) 3D L2 edge lengths."""
    verts = np.asarray(verts)
    if verts.ndim == 2:
        verts = verts[None]
    idx_a = np.asarray([e[0] for e in edges])
    idx_b = np.asarray([e[1] for e in edges])
    diff = verts[:, idx_a, :] - verts[:, idx_b, :]  # (N, 24, 3)
    return np.linalg.norm(diff, axis=-1)  # (N, 24)


def compute_edge_lengths_2d(verts: np.ndarray, edges: list[tuple[int, int]]) -> np.ndarray:
    """(N, 14, 3) → (N, 24) 2D pixel-distance edge lengths (orthographic projection: XY only)."""
    verts = np.asarray(verts)
    if verts.ndim == 2:
        verts = verts[None]
    idx_a = np.asarray([e[0] for e in edges])
    idx_b = np.asarray([e[1] for e in edges])
    diff = verts[:, idx_a, :2] - verts[:, idx_b, :2]  # (N, 24, 2)
    return np.linalg.norm(diff, axis=-1)


def compute_metrics(pred: np.ndarray, gt: np.ndarray) -> dict:
    """Edge length metrics. pred/gt both (N, 24).

    Returns aggregated + per-edge MAE + per-instance error vectors.
    """
    pred = np.asarray(pred, dtype=np.float64)
    gt = np.asarray(gt, dtype=np.float64)
    assert pred.shape == gt.shape, f"shape mismatch: {pred.shape} vs {gt.shape}"
    assert pred.shape[1] == 24, f"expected 24 edges, got {pred.shape[1]}"

    abs_err = np.abs(pred - gt)                        # (N, 24)
    sq_err = (pred - gt) ** 2
    # Avoid division by zero in MAPE — fall back to abs_err where gt == 0
    safe_gt = np.where(np.abs(gt) > 1e-9, gt, 1.0)
    pct_err = np.where(np.abs(gt) > 1e-9, abs_err / np.abs(safe_gt) * 100.0, np.nan)

    per_inst_mae = abs_err.mean(axis=1)                 # (N,)
    per_inst_rmse = np.sqrt(sq_err.mean(axis=1))
    per_inst_mape = np.nanmean(pct_err, axis=1)

    # Shape σ: pred 자체에서 24 edge 길이의 std (per-instance), 평균 — shape consistency 지표
    pred_shape_sigma = pred.std(axis=1).mean()        # avg of per-instance std of 24 edges
    gt_shape_sigma = gt.std(axis=1).mean()            # GT는 ~0 (rhombic-dodec invariant)
    return {
        "mae": float(abs_err.mean()),
        "rmse": float(np.sqrt(sq_err.mean())),
        "mape": float(np.nanmean(pct_err)),
        "median_mape": float(np.nanmedian(pct_err)),
        "shape_sigma_pred": float(pred_shape_sigma),  # per-instance edge-len std → 평균
        "shape_sigma_gt": float(gt_shape_sigma),
        "per_edge_mae": abs_err.mean(axis=0).tolist(),                  # (24,)
        "per_edge_mape": np.nanmean(pct_err, axis=0).tolist(),          # (24,)
        "per_instance_mae": per_inst_mae.tolist(),                      # (N,)
        "per_instance_rmse": per_inst_rmse.tolist(),
        "per_instance_mape": per_inst_mape.tolist(),
        "n": int(pred.shape[0]),
    }
