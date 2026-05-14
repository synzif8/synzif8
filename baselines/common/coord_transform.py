"""Option-D coordinate transforms: world ↔ crop-local normalized.

All candidates output in crop-local normalized space; conversion happens at
data-loading time (target prep) and inference time (metric/viz).

Convention
----------
- crop_affine: (2, 3) tensor mapping world (X, Y) → crop pixel (in 256 frame).
- crop_size  : H = W of input crop (default 256).
- pixel_norm = (pixel - cs/2) / (cs/2)  ∈ [-1, 1]  when pixel ∈ [0, cs].
- t_z_norm   = (t_z - tz_mean) / tz_std
- s_norm     = (s   - s_mean)  / s_std
"""

from __future__ import annotations

import json
import os
from functools import lru_cache

import numpy as np
import torch

DEFAULT_STATS = "<PROJECT_ROOT>/baselines/phase0_common/z_scale_stats.json"


@lru_cache(maxsize=1)
def load_stats(path: str = DEFAULT_STATS) -> dict:
    with open(path) as f:
        return json.load(f)


# ---------- numpy variants (data loading / eval) ----------


def world_to_crop_t_np(
    t_world: np.ndarray,            # (3,) or (B, 3)
    s_world: np.ndarray | float,    # () or (B,)
    crop_affine: np.ndarray,        # (2, 3) or (B, 2, 3)
    crop_size: int,
    stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Returns (t_norm (..., 3), s_norm (...,))."""
    stats = stats or load_stats()
    t_world = np.asarray(t_world, dtype=np.float32)
    s_world = np.asarray(s_world, dtype=np.float32)
    crop_affine = np.asarray(crop_affine, dtype=np.float32)
    half = crop_size / 2.0

    # xy → crop pixel via affine
    if t_world.ndim == 1:
        xy_pix = crop_affine[:, :2] @ t_world[:2] + crop_affine[:, 2]
    else:
        xy_pix = np.einsum("bij,bj->bi", crop_affine[:, :, :2], t_world[..., :2]) + crop_affine[:, :, 2]
    xy_norm = (xy_pix - half) / half

    z_norm = (t_world[..., 2] - stats["tz_mean"]) / stats["tz_std"]
    t_norm = np.concatenate([xy_norm, z_norm[..., None]], axis=-1).astype(np.float32)
    s_norm = ((s_world - stats["scale_mean"]) / stats["scale_std"]).astype(np.float32)
    return t_norm, s_norm


def crop_to_world_t_np(
    t_norm: np.ndarray,             # (..., 3)
    s_norm: np.ndarray | float,     # (...,)
    crop_affine: np.ndarray,        # (..., 2, 3)
    crop_size: int,
    stats: dict | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Inverse of world_to_crop_t_np."""
    stats = stats or load_stats()
    t_norm = np.asarray(t_norm, dtype=np.float32)
    s_norm = np.asarray(s_norm, dtype=np.float32)
    crop_affine = np.asarray(crop_affine, dtype=np.float32)
    half = crop_size / 2.0

    xy_pix = t_norm[..., :2] * half + half
    A = crop_affine[..., :2]
    b = crop_affine[..., 2]
    if A.ndim == 2:
        A_inv = np.linalg.inv(A)
        xy_world = (xy_pix - b) @ A_inv.T
    else:
        A_inv = np.linalg.inv(A)
        xy_world = np.einsum("bij,bj->bi", A_inv, xy_pix - b)
    tz = t_norm[..., 2] * stats["tz_std"] + stats["tz_mean"]
    t_world = np.concatenate([xy_world, tz[..., None]], axis=-1).astype(np.float32)
    s_world = (s_norm * stats["scale_std"] + stats["scale_mean"]).astype(np.float32)
    return t_world, s_world


def world_to_crop_kp3d_np(
    kp_world: np.ndarray,           # (V, 3) or (B, V, 3)
    crop_affine: np.ndarray,        # (2, 3) or (B, 2, 3)
    crop_size: int,
    stats: dict | None = None,
) -> np.ndarray:
    """Per-vertex normalize for B-category kp3d output."""
    stats = stats or load_stats()
    kp_world = np.asarray(kp_world, dtype=np.float32)
    crop_affine = np.asarray(crop_affine, dtype=np.float32)
    half = crop_size / 2.0

    if kp_world.ndim == 2:
        xy_pix = kp_world[:, :2] @ crop_affine[:, :2].T + crop_affine[:, 2]
    else:
        xy_pix = np.einsum("bij,bvj->bvi", crop_affine[:, :, :2], kp_world[..., :2]) + crop_affine[:, None, :, 2]
    xy_norm = (xy_pix - half) / half
    z_norm = (kp_world[..., 2] - stats["tz_mean"]) / stats["tz_std"]
    return np.concatenate([xy_norm, z_norm[..., None]], axis=-1).astype(np.float32)


def crop_to_world_kp3d_np(
    kp_norm: np.ndarray,            # (..., V, 3)
    crop_affine: np.ndarray,        # (..., 2, 3)
    crop_size: int,
    stats: dict | None = None,
) -> np.ndarray:
    stats = stats or load_stats()
    kp_norm = np.asarray(kp_norm, dtype=np.float32)
    crop_affine = np.asarray(crop_affine, dtype=np.float32)
    half = crop_size / 2.0

    xy_pix = kp_norm[..., :2] * half + half
    A = crop_affine[..., :2]
    b = crop_affine[..., 2]
    if A.ndim == 2:
        A_inv = np.linalg.inv(A)
        xy_world = (xy_pix - b) @ A_inv.T
    else:
        A_inv = np.linalg.inv(A)
        xy_world = np.einsum("bij,bvj->bvi", A_inv, xy_pix - b[:, None, :])
    z_world = kp_norm[..., 2] * stats["tz_std"] + stats["tz_mean"]
    return np.concatenate([xy_world, z_world[..., None]], axis=-1).astype(np.float32)


# ---------- torch variants (loss / eval differentiable) ----------


def world_to_crop_t_torch(
    t_world: torch.Tensor,          # (B, 3)
    s_world: torch.Tensor,          # (B,)
    crop_affine: torch.Tensor,      # (B, 2, 3)
    crop_size: int,
    stats: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    stats = stats or load_stats()
    half = crop_size / 2.0
    A = crop_affine[:, :, :2]; b = crop_affine[:, :, 2]
    xy_pix = torch.einsum("bij,bj->bi", A, t_world[..., :2]) + b
    xy_norm = (xy_pix - half) / half
    z_norm = (t_world[..., 2] - stats["tz_mean"]) / stats["tz_std"]
    t_norm = torch.cat([xy_norm, z_norm.unsqueeze(-1)], dim=-1)
    s_norm = (s_world - stats["scale_mean"]) / stats["scale_std"]
    return t_norm.to(t_world.dtype), s_norm.to(s_world.dtype)


def crop_to_world_t_torch(
    t_norm: torch.Tensor,
    s_norm: torch.Tensor,
    crop_affine: torch.Tensor,
    crop_size: int,
    stats: dict | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    stats = stats or load_stats()
    half = crop_size / 2.0
    A = crop_affine[:, :, :2]; b = crop_affine[:, :, 2]
    xy_pix = t_norm[..., :2] * half + half
    A_inv = torch.linalg.inv(A)
    xy_world = torch.einsum("bij,bj->bi", A_inv, xy_pix - b)
    tz = t_norm[..., 2] * stats["tz_std"] + stats["tz_mean"]
    t_world = torch.cat([xy_world, tz.unsqueeze(-1)], dim=-1)
    s_world = s_norm * stats["scale_std"] + stats["scale_mean"]
    return t_world.to(t_norm.dtype), s_world.to(s_norm.dtype)


def world_to_crop_kp3d_torch(
    kp_world: torch.Tensor,         # (B, V, 3)
    crop_affine: torch.Tensor,      # (B, 2, 3)
    crop_size: int,
    stats: dict | None = None,
) -> torch.Tensor:
    stats = stats or load_stats()
    half = crop_size / 2.0
    kp_world = kp_world.float()
    crop_affine = crop_affine.float()
    A = crop_affine[:, :, :2]; b = crop_affine[:, :, 2]
    xy_pix = torch.einsum("bij,bvj->bvi", A, kp_world[..., :2]) + b.unsqueeze(1)
    xy_norm = (xy_pix - half) / half
    z_norm = (kp_world[..., 2] - stats["tz_mean"]) / stats["tz_std"]
    return torch.cat([xy_norm, z_norm.unsqueeze(-1)], dim=-1)


def crop_to_world_kp3d_torch(
    kp_norm: torch.Tensor,          # (B, V, 3)
    crop_affine: torch.Tensor,      # (B, 2, 3)
    crop_size: int,
    stats: dict | None = None,
) -> torch.Tensor:
    stats = stats or load_stats()
    half = crop_size / 2.0
    kp_norm = kp_norm.float()
    crop_affine = crop_affine.float()
    A = crop_affine[:, :, :2]; b = crop_affine[:, :, 2]
    xy_pix = kp_norm[..., :2] * half + half
    A_inv = torch.linalg.inv(A)
    xy_world = torch.einsum("bij,bvj->bvi", A_inv, xy_pix - b.unsqueeze(1))
    z_world = kp_norm[..., 2] * stats["tz_std"] + stats["tz_mean"]
    return torch.cat([xy_world, z_world.unsqueeze(-1)], dim=-1)
