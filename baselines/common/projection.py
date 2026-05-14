"""Orthographic projection helpers."""

from __future__ import annotations

import numpy as np
import torch


def ortho_project_np(points_3d: np.ndarray, crop_affine: np.ndarray) -> np.ndarray:
    """points_3d: (..., 3). crop_affine: (2, 3). Returns (..., 2) crop-pixel coords."""
    p2d = points_3d[..., :2]
    return p2d @ crop_affine[:, :2].T + crop_affine[:, 2]


def ortho_project_torch(points_3d: torch.Tensor, crop_affine: torch.Tensor) -> torch.Tensor:
    """Batched: points_3d (B, N, 3), crop_affine (B, 2, 3). Returns (B, N, 2)."""
    p2d = points_3d[..., :2]
    A = crop_affine[..., :2]  # (B, 2, 2)
    b = crop_affine[..., 2]   # (B, 2)
    return torch.einsum("bij,bnj->bni", A, p2d) + b.unsqueeze(1)


def apply_pose_np(
    canonical_verts: np.ndarray, R: np.ndarray, t: np.ndarray, s: float
) -> np.ndarray:
    """s * canonical_verts @ R^T + t."""
    return s * canonical_verts @ R.T + t


def apply_pose_torch(
    canonical_verts: torch.Tensor, R: torch.Tensor, t: torch.Tensor, s: torch.Tensor
) -> torch.Tensor:
    """Batched. canonical (V,3) or (B,V,3); R (B,3,3); t (B,3); s (B,)."""
    if canonical_verts.dim() == 2:
        canonical_verts = canonical_verts.unsqueeze(0).expand(R.shape[0], -1, -1)
    out = torch.einsum("bij,bnj->bni", R, canonical_verts) * s.view(-1, 1, 1)
    out = out + t.view(-1, 1, 3)
    return out
