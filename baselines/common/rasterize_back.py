"""Back-surface rasterizer — same as front NOCS but min-z wins.

create_dataset_v6.py renders with painter's algorithm where larger ``z`` = closer
to camera. Front (visible) surface = max-z wins. Back (occluded) surface = min-z
wins. Used by HccePose's front/back code supervision.
"""

from __future__ import annotations

import numpy as np

from common.rasterize_nocs import FACES


def _rasterize_triangle_attr_min_z(
    v2d: np.ndarray,
    vz: np.ndarray,
    v_attr: np.ndarray,
    attr_canvas: np.ndarray,
    z_buffer: np.ndarray,
    valid_canvas: np.ndarray,
) -> None:
    H, W = z_buffer.shape
    x_min = max(0, int(np.floor(v2d[:, 0].min())))
    y_min = max(0, int(np.floor(v2d[:, 1].min())))
    x_max = min(W - 1, int(np.ceil(v2d[:, 0].max())))
    y_max = min(H - 1, int(np.ceil(v2d[:, 1].max())))
    if x_max < x_min or y_max < y_min:
        return
    x0, y0 = v2d[0]; x1, y1 = v2d[1]; x2, y2 = v2d[2]
    denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    if abs(denom) < 1e-9:
        return
    ys, xs = np.mgrid[y_min:y_max + 1, x_min:x_max + 1]
    xs_f = xs.astype(np.float32)
    ys_f = ys.astype(np.float32)
    l0 = ((y1 - y2) * (xs_f - x2) + (x2 - x1) * (ys_f - y2)) / denom
    l1 = ((y2 - y0) * (xs_f - x2) + (x0 - x2) * (ys_f - y2)) / denom
    l2 = 1.0 - l0 - l1
    inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
    z_tri = l0 * vz[0] + l1 * vz[1] + l2 * vz[2]
    ztile = z_buffer[y_min:y_max + 1, x_min:x_max + 1]
    mask = inside & (z_tri < ztile)  # min wins (back)
    if not mask.any():
        return
    attr_tile = attr_canvas[y_min:y_max + 1, x_min:x_max + 1]
    for c in range(v_attr.shape[1]):
        a = l0 * v_attr[0, c] + l1 * v_attr[1, c] + l2 * v_attr[2, c]
        attr_tile[..., c][mask] = a[mask]
    z_buffer[y_min:y_max + 1, x_min:x_max + 1][mask] = z_tri[mask]
    valid_canvas[y_min:y_max + 1, x_min:x_max + 1][mask] = True


def render_back_canonical_crop(
    canonical_verts: np.ndarray,
    vertices_cam: np.ndarray,
    crop_affine: np.ndarray,
    crop_size: int,
    canonical_radius: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Render per-pixel back-surface NOCS in [-1, 1]."""
    verts_2d = vertices_cam[:, :2] @ crop_affine[:, :2].T + crop_affine[:, 2]
    verts_z = vertices_cam[:, 2].astype(np.float32)
    nocs_canonical = (canonical_verts / canonical_radius).astype(np.float32)
    H = W = crop_size
    attr = np.zeros((H, W, 3), dtype=np.float32)
    z_buf = np.full((H, W), np.inf, dtype=np.float32)
    valid = np.zeros((H, W), dtype=bool)
    for face in FACES:
        v = np.asarray(face, dtype=np.int64)
        v2d = verts_2d[v].astype(np.float32)
        vz = verts_z[v]
        vattr = nocs_canonical[v]
        for tri in ((0, 1, 2), (0, 2, 3)):
            tri_idx = np.asarray(tri, dtype=np.int64)
            _rasterize_triangle_attr_min_z(
                v2d=v2d[tri_idx], vz=vz[tri_idx], v_attr=vattr[tri_idx],
                attr_canvas=attr, z_buffer=z_buf, valid_canvas=valid,
            )
    return attr, valid


def hcce_encode_normalized(norm_xyz: np.ndarray, bits: int = 8) -> np.ndarray:
    """norm_xyz (H, W, 3) ∈ [0, 1] → (H, W, 3*bits) binary code via paper Eq.

    bit_b of dim d = floor(c_d * 2^(b+1)) mod 2  (b = 0..bits-1).
    """
    H, W, _ = norm_xyz.shape
    out = np.zeros((H, W, 3 * bits), dtype=np.float32)
    for d in range(3):
        c = np.clip(norm_xyz[..., d], 0.0, 1.0 - 1e-6)
        for b in range(bits):
            out[..., d * bits + b] = (np.floor(c * (2 ** (b + 1))).astype(np.int64) % 2).astype(np.float32)
    return out


def hcce_decode_to_normalized(code: np.ndarray, bits: int = 8) -> np.ndarray:
    """Inverse of hcce_encode_normalized.

    code: (..., 3*bits) ∈ {0, 1} → norm xyz (..., 3) ∈ [0, 1].
    """
    *prefix, C = code.shape
    assert C == 3 * bits
    code = code.reshape(*prefix, 3, bits)
    weights = np.asarray([2.0 ** -(b + 1) for b in range(bits)], dtype=np.float32)
    return (code * weights).sum(axis=-1)
