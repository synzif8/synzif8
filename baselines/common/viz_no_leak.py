"""GT-leak-free visualization utilities.

Design contract
---------------
Every render function takes ONLY the quantities it needs to draw that one panel:
- `render_mesh_from_pose(image, R, t, s, canonical_verts, crop_affine, faces, color)`
  — renders a mesh from a single pose. Cannot see GT.
- `render_kp3d_projection(image, kp_3d, crop_affine, color)`
  — projects a single set of 3D keypoints. Cannot see GT.
- `render_mask_error(mask_pred, mask_gt)` — per-pixel mask comparison; allowed
  because the inputs are already rasterized images, not geometry. No pose leak
  is possible.
- `build_2x2(panel_tl, panel_tr, panel_bl, panel_br, labels)` — pure ndarray
  compositor. Doesn't see poses at all.

Callers must invoke render_* for GT and prediction separately, assembling the
grid at the very end. The signatures disallow a GT-pose parameter to enter
a pred-render function.

Pytest enforcement lives in lineup/tests/test_viz_no_leak.py.
"""

from __future__ import annotations

from typing import Sequence

import cv2
import numpy as np

# rhombic-dodecahedron face topology (matches create_dataset_v6.py)
FACES: list[list[int]] = [
    [8, 5, 1, 6], [8, 6, 2, 9], [8, 9, 4, 7], [8, 7, 3, 5],
    [0, 10, 1, 11], [0, 11, 2, 13], [0, 13, 4, 12], [0, 12, 3, 10],
    [1, 5, 3, 10], [3, 7, 4, 12], [4, 9, 2, 13], [2, 6, 1, 11],
]


def render_mesh_from_pose(
    image: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    s: float,
    canonical_verts: np.ndarray,
    crop_affine: np.ndarray,
    faces: Sequence[Sequence[int]] = FACES,
    color: tuple[int, int, int] = (0, 255, 0),
    thickness: int = 1,
    vertex_radius: int = 2,
) -> np.ndarray:
    """Draw mesh from (R, t, s). Independent of any ground-truth inputs.

    Args:
        image: (H, W, 3) uint8 BGR canvas.
        R: (3, 3) rotation, object → camera.
        t: (3,) translation.
        s: uniform scale (canonical → camera).
        canonical_verts: (V, 3) canonical object vertices.
        crop_affine: (2, 3) mapping camera (X, Y) → crop pixels.
        faces: index tuples per face.
        color: BGR line color.
        thickness: polyline thickness.
        vertex_radius: vertex dot radius.
    """
    canonical_verts = np.asarray(canonical_verts, dtype=np.float32)
    R = np.asarray(R, dtype=np.float32).reshape(3, 3)
    t = np.asarray(t, dtype=np.float32).reshape(3)
    crop_affine = np.asarray(crop_affine, dtype=np.float32).reshape(2, 3)

    verts_cam = float(s) * canonical_verts @ R.T + t
    verts_2d = (verts_cam[:, :2] @ crop_affine[:, :2].T) + crop_affine[:, 2]
    out = image.copy()
    for face in faces:
        poly = verts_2d[list(face)].astype(np.int32)
        cv2.polylines(out, [poly], isClosed=True, color=color, thickness=thickness)
    for p in verts_2d.astype(np.int32):
        cv2.circle(out, (int(p[0]), int(p[1])), vertex_radius, color, -1)
    return out


def render_kp3d_projection(
    image: np.ndarray,
    kp_3d: np.ndarray,
    crop_affine: np.ndarray,
    color: tuple[int, int, int] = (0, 255, 255),
    radius: int = 3,
) -> np.ndarray:
    """Project N×3 keypoints (orthographic) and draw them on ``image``.

    Args:
        image: (H, W, 3) uint8 canvas.
        kp_3d: (N, 3) camera-frame keypoints.
        crop_affine: (2, 3).
    """
    kp_3d = np.asarray(kp_3d, dtype=np.float32).reshape(-1, 3)
    crop_affine = np.asarray(crop_affine, dtype=np.float32).reshape(2, 3)
    kp_2d = (kp_3d[:, :2] @ crop_affine[:, :2].T) + crop_affine[:, 2]
    out = image.copy()
    for p in kp_2d.astype(np.int32):
        cv2.circle(out, (int(p[0]), int(p[1])), radius, color, -1)
    return out


def render_mask_error(mask_pred: np.ndarray, mask_gt: np.ndarray) -> np.ndarray:
    """Per-pixel mask error map. Inputs are already-rasterized images — no pose leak possible."""
    mp = mask_pred.astype(bool)
    mg = mask_gt.astype(bool)
    assert mp.shape == mg.shape, f"mask shape mismatch {mp.shape} vs {mg.shape}"
    H, W = mp.shape
    out = np.zeros((H, W, 3), np.uint8)
    out[mp & ~mg] = (0, 0, 255)   # red: pred-only (false positive)
    out[mg & ~mp] = (255, 0, 0)   # blue: gt-only (false negative)
    out[mp & mg] = (0, 128, 0)    # green: overlap (true positive)
    return out


def build_2x2(
    panel_tl: np.ndarray,
    panel_tr: np.ndarray,
    panel_bl: np.ndarray,
    panel_br: np.ndarray,
    labels: tuple[str, str, str, str] = ("input", "gt", "pred", "error"),
) -> np.ndarray:
    shapes = {p.shape for p in (panel_tl, panel_tr, panel_bl, panel_br)}
    assert len(shapes) == 1, f"panel shapes differ: {shapes}"

    def _label(im: np.ndarray, txt: str) -> np.ndarray:
        out = im.copy()
        w = min(len(txt) * 10 + 12, out.shape[1])
        cv2.rectangle(out, (0, 0), (w, 24), (0, 0, 0), -1)
        cv2.putText(out, txt, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
        return out

    top = np.concatenate([_label(panel_tl, labels[0]), _label(panel_tr, labels[1])], axis=1)
    bot = np.concatenate([_label(panel_bl, labels[2]), _label(panel_br, labels[3])], axis=1)
    return np.concatenate([top, bot], axis=0)


class LeakGuard:
    """Runtime wrapper that forbids GT-typed keyword args on pred-render callees."""

    def __init__(self, forbidden_prefixes: tuple[str, ...] = ("gt_", "target_", "ground_truth_")):
        self.forbidden = forbidden_prefixes

    def __call__(self, fn):
        def wrapped(*args, **kwargs):
            for k in kwargs:
                if any(k.startswith(p) for p in self.forbidden):
                    raise RuntimeError(f"GT leak: {fn.__name__} received kwarg {k!r}")
            return fn(*args, **kwargs)
        wrapped.__name__ = fn.__name__
        return wrapped
