"""Numpy triangle rasterizer for NOCS / surface-attribute maps.

Orthographic projection only. Uses max-z-wins z-buffer (matches
create_dataset_v6.py convention where larger ``z_depth`` ⇒ closer to camera).
"""

from __future__ import annotations

import numpy as np

FACES: list[list[int]] = [
    [8, 5, 1, 6], [8, 6, 2, 9], [8, 9, 4, 7], [8, 7, 3, 5],
    [0, 10, 1, 11], [0, 11, 2, 13], [0, 13, 4, 12], [0, 12, 3, 10],
    [1, 5, 3, 10], [3, 7, 4, 12], [4, 9, 2, 13], [2, 6, 1, 11],
]


def rasterize_triangle_attr(
    v2d: np.ndarray,        # (3, 2) float
    vz: np.ndarray,          # (3,) float depth (max wins)
    v_attr: np.ndarray,      # (3, C) float per-vertex attribute
    attr_canvas: np.ndarray,  # (H, W, C) out — updated in place
    z_buffer: np.ndarray,    # (H, W) float — updated in place
    valid_canvas: np.ndarray,  # (H, W) bool — updated in place
) -> None:
    """Rasterize one triangle with per-vertex attribute; max-z wins."""
    H, W = z_buffer.shape
    x_min = max(0, int(np.floor(v2d[:, 0].min())))
    y_min = max(0, int(np.floor(v2d[:, 1].min())))
    x_max = min(W - 1, int(np.ceil(v2d[:, 0].max())))
    y_max = min(H - 1, int(np.ceil(v2d[:, 1].max())))
    if x_max < x_min or y_max < y_min:
        return

    x0, y0 = v2d[0]
    x1, y1 = v2d[1]
    x2, y2 = v2d[2]
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
    mask = inside & (z_tri > ztile)  # max wins
    if not mask.any():
        return
    # interpolate attributes
    attr_tile = attr_canvas[y_min:y_max + 1, x_min:x_max + 1]
    for c in range(v_attr.shape[1]):
        a = l0 * v_attr[0, c] + l1 * v_attr[1, c] + l2 * v_attr[2, c]
        attr_tile[..., c][mask] = a[mask]
    z_buffer[y_min:y_max + 1, x_min:x_max + 1][mask] = z_tri[mask]
    valid_canvas[y_min:y_max + 1, x_min:x_max + 1][mask] = True


def render_nocs_crop(
    canonical_verts: np.ndarray,   # (V, 3)
    vertices_cam: np.ndarray,      # (V, 3) world = camera frame
    crop_affine: np.ndarray,       # (2, 3)
    crop_size: int,
    faces: list[list[int]] = FACES,
    canonical_radius: float = 2.0,  # |max canonical coord| for normalization
) -> tuple[np.ndarray, np.ndarray]:
    """Render per-pixel (NOCS, valid_mask) in crop coordinates.

    NOCS = canonical / canonical_radius ∈ [-1, 1] (3 channels).
    """
    verts_2d = vertices_cam[:, :2] @ crop_affine[:, :2].T + crop_affine[:, 2]
    verts_z = vertices_cam[:, 2].astype(np.float32)
    nocs_canonical = (canonical_verts / canonical_radius).astype(np.float32)

    H = W = crop_size
    attr_canvas = np.zeros((H, W, 3), dtype=np.float32)
    z_buffer = np.full((H, W), -np.inf, dtype=np.float32)
    valid = np.zeros((H, W), dtype=bool)

    for face in faces:
        v = np.asarray(face, dtype=np.int64)
        v2d = verts_2d[v].astype(np.float32)
        vz = verts_z[v]
        vattr = nocs_canonical[v]
        # split quad into two triangles
        for tri in ((0, 1, 2), (0, 2, 3)):
            tri_idx = np.asarray(tri, dtype=np.int64)
            rasterize_triangle_attr(
                v2d=v2d[tri_idx],
                vz=vz[tri_idx],
                v_attr=vattr[tri_idx],
                attr_canvas=attr_canvas,
                z_buffer=z_buffer,
                valid_canvas=valid,
            )
    return attr_canvas, valid


def nocs_to_rgb(nocs: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """Visualize NOCS: map [-1, 1] -> [0, 255] RGB, black where invalid."""
    rgb = ((nocs + 1.0) * 0.5 * 255.0).clip(0, 255).astype(np.uint8)
    rgb[~valid] = 0
    return rgb
