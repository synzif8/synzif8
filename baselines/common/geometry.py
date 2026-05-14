"""Geometry helpers — O_24 symmetry group, Umeyama/Kabsch, symmetry-aware metrics.

Rhombic dodecahedron shares its proper rotation group with the cube / octahedron:
|O| = 24 signed-permutation matrices with det = +1. These rotations preserve both
the canonical vertex set ``base_vertices`` (axial ±2 points + all ±1 corners).
"""

from __future__ import annotations

from itertools import permutations, product

import numpy as np
import torch


def O24_rotations() -> np.ndarray:
    """Return (24, 3, 3) proper rotation matrices of the octahedral group."""
    I = np.eye(3, dtype=np.float64)
    mats = []
    for perm in permutations(range(3)):
        P = I[:, list(perm)]
        for signs in product((1, -1), repeat=3):
            M = P @ np.diag(signs).astype(np.float64)
            if np.linalg.det(M) > 0.5:
                mats.append(M)
    M = np.stack(mats, axis=0).astype(np.float32)
    assert M.shape == (24, 3, 3)
    return M


_O24 = O24_rotations()


def umeyama_np(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """(R, t, s) with dst ≈ s * R @ src + t. src/dst: (N, 3)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    mu_s = src.mean(axis=0)
    mu_d = dst.mean(axis=0)
    sc = src - mu_s
    dc = dst - mu_d
    var_s = (sc ** 2).sum() / n
    cov = dc.T @ sc / n
    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt
    s = float((D * np.diag(S)).sum() / var_s)
    t = mu_d - s * R @ mu_s
    return R.astype(np.float32), t.astype(np.float32), float(s)


def umeyama_torch(
    src: torch.Tensor, dst: torch.Tensor, allow_scale: bool = True
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Batched Umeyama. src, dst: (B, N, 3). Returns (R, t, s) with shapes (B,3,3), (B,3), (B,)."""
    assert src.shape == dst.shape, f"{src.shape} vs {dst.shape}"
    B, N, _ = src.shape
    mu_s = src.mean(dim=1, keepdim=True)  # (B, 1, 3)
    mu_d = dst.mean(dim=1, keepdim=True)
    sc = src - mu_s
    dc = dst - mu_d
    var_s = (sc ** 2).sum(dim=(1, 2)) / N  # (B,)
    cov = torch.einsum("bni,bnj->bij", dc, sc) / N  # (B, 3, 3)
    U, D, Vh = torch.linalg.svd(cov)
    det = torch.det(U @ Vh)
    S = torch.eye(3, device=src.device).repeat(B, 1, 1)
    S[:, 2, 2] = torch.sign(det)
    R = U @ S @ Vh  # (B, 3, 3)
    if allow_scale:
        s = (D * torch.diagonal(S, dim1=-2, dim2=-1)).sum(dim=-1) / var_s
    else:
        s = torch.ones(B, device=src.device, dtype=src.dtype)
    # t = mu_d - s * R @ mu_s
    t = mu_d.squeeze(1) - s.unsqueeze(-1) * torch.einsum("bij,bj->bi", R, mu_s.squeeze(1))
    return R, t, s


def sym_aware_vertex_error(
    pred_verts_cam: np.ndarray,
    gt_verts_cam: np.ndarray,
    canonical_verts: np.ndarray,
    gt_R: np.ndarray,
    gt_t: np.ndarray,
    gt_s: float,
    symmetries: np.ndarray | None = None,
) -> dict:
    """ADD-S-like metric with O24 symmetry.

    For each symmetry G in O_24, compute the predicted reprojection using
    (R @ G, t, s) — i.e. relabel canonical vertices. Return minimum mean-L2.

    Returns dict with keys:
        add_s_min : float (WU, mean per-vertex L2 under best symmetry)
        best_sym_idx : int
    """
    if symmetries is None:
        symmetries = _O24
    pred_verts_cam = np.asarray(pred_verts_cam, dtype=np.float32)
    canonical_verts = np.asarray(canonical_verts, dtype=np.float32)
    errs = []
    for G in symmetries:
        sym_canonical = canonical_verts @ G.T  # equivalent to applying G to canonical labels
        gt_sym = gt_s * sym_canonical @ gt_R.T + gt_t
        err = np.linalg.norm(pred_verts_cam - gt_sym, axis=1).mean()
        errs.append(err)
    errs = np.asarray(errs)
    return {"add_s_min": float(errs.min()), "best_sym_idx": int(errs.argmin())}


def sym_aware_rot_error_deg(
    R_pred: np.ndarray, R_gt: np.ndarray, symmetries: np.ndarray | None = None
) -> float:
    """Geodesic rotation error in degrees under O_24 symmetry."""
    if symmetries is None:
        symmetries = _O24
    best = 180.0
    for G in symmetries:
        Rd = R_pred @ (R_gt @ G).T  # R_pred * (R_gt G)^-1
        tr = np.clip((np.trace(Rd) - 1) * 0.5, -1.0, 1.0)
        ang = np.degrees(np.arccos(tr))
        if ang < best:
            best = ang
    return float(best)


def procrustes_from_kp3d(
    pred_kp_3d: np.ndarray, canonical_verts: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float]:
    """Decode (R, t, s) from predicted 3D keypoints via Umeyama with canonical."""
    return umeyama_np(canonical_verts, pred_kp_3d)
