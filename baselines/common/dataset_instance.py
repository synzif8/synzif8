"""Phase 0.2 — InstanceSample dataset.

Produces per-instance crops + full geometry for any downstream baseline.
The camera is **pure orthographic** (create_dataset_v6.py: `pts_2d = verts[:,:2]`),
so projection is just ``(x_pix, y_pix) = (X_world, Y_world)``. After cropping,
the model sees coordinates in a rescaled local frame, and the `crop_affine`
field encodes the 2×3 affine that takes world (x, y) → crop pixels.

Schema (per item)
-----------------
img_crop       : FloatTensor [1, H, W]   grayscale, range [-1, 1]
mask_visible   : BoolTensor  [1, H, W]   modal mask in crop
mask_amodal    : BoolTensor  [1, H, W]   amodal mask in crop
R              : FloatTensor [3, 3]      object → world rotation
t              : FloatTensor [3]         translation (world frame = camera frame)
scale          : FloatTensor []          uniform scale
K_ortho        : FloatTensor [4]         (sx, sy, cx_shift, cy_shift) of crop affine
bbox_full      : FloatTensor [4]         (x0, y0, x1, y1) in original 1024×686 image
crop_affine    : FloatTensor [2, 3]      full-pixel (X, Y) → crop (px, py)
vertices_obj   : FloatTensor [14, 3]     canonical (base_vertices)
vertices_cam   : FloatTensor [14, 3]     world-frame (= camera-frame for ortho)
kp_2d_crop     : FloatTensor [14, 2]     vertex projection in crop pixels
visibility     : FloatTensor []          ratio from metadata
obj_id         : LongTensor  []
scene_id       : LongTensor  []
"""

from __future__ import annotations

import json
import os
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

DATASET_DIR_DEFAULT = "<DATA_ROOT>/dataset_v6"
POSE_CACHE_DIR_DEFAULT = "<PROJECT_ROOT>/baselines/phase0_common/pose_cache"


def _load_split(split_path: str) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    with open(split_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            assert len(parts) == 2, f"bad split line: {line!r}"
            out.append((int(parts[0]), int(parts[1])))
    return out


def _crop_square(
    img: np.ndarray, bbox_xyxy: tuple[int, int, int, int], pad_ratio: float
) -> tuple[np.ndarray, tuple[int, int, int, int], float]:
    """Return (cropped_uncropped_size, bbox_after_pad_on_full_image, side).

    Produces a square region on the original image with the given pad ratio,
    zero-padding any out-of-image area. Returns the raw (unresized) crop.
    """
    H_img, W_img = img.shape[:2]
    x0, y0, x1, y1 = bbox_xyxy
    w = x1 - x0
    h = y1 - y0
    side = max(w, h)
    pad = side * pad_ratio
    side_padded = int(round(side + 2 * pad))
    cx = (x0 + x1) / 2.0
    cy = (y0 + y1) / 2.0
    bx0 = int(round(cx - side_padded / 2.0))
    by0 = int(round(cy - side_padded / 2.0))
    bx1 = bx0 + side_padded
    by1 = by0 + side_padded

    pad_l = max(0, -bx0)
    pad_t = max(0, -by0)
    pad_r = max(0, bx1 - W_img)
    pad_b = max(0, by1 - H_img)

    if img.ndim == 2:
        pad_widths = ((pad_t, pad_b), (pad_l, pad_r))
    else:
        pad_widths = ((pad_t, pad_b), (pad_l, pad_r), (0, 0))
    img_padded = np.pad(img, pad_widths, mode="constant")

    sx0 = bx0 + pad_l
    sy0 = by0 + pad_t
    crop = img_padded[sy0 : sy0 + side_padded, sx0 : sx0 + side_padded]
    return crop, (bx0, by0, bx1, by1), float(side_padded)


class InstanceDataset(Dataset):
    """Dataset of rhombic-dodecahedron instances from dataset_v6."""

    def __init__(
        self,
        split_path: str,
        dataset_dir: str = DATASET_DIR_DEFAULT,
        pose_cache_dir: str = POSE_CACHE_DIR_DEFAULT,
        crop_size: int = 256,
        pad_ratio: float = 0.15,
        use_styled: bool = True,
    ) -> None:
        self.entries = _load_split(split_path)
        self.dataset_dir = dataset_dir
        self.pose_cache_dir = pose_cache_dir
        self.crop_size = int(crop_size)
        self.pad_ratio = float(pad_ratio)
        self.use_styled = bool(use_styled)

    def __len__(self) -> int:
        return len(self.entries)

    def _paths(self, scene_id: int, obj_id: int) -> dict[str, str]:
        sdir = self.dataset_dir
        img_name = (
            f"render_{scene_id:04d}_styled.png"
            if self.use_styled
            else f"render_{scene_id:04d}.png"
        )
        return {
            "img": os.path.join(sdir, img_name),
            "amodal": os.path.join(sdir, f"render_{scene_id:04d}_masks", f"obj_{obj_id:04d}_amodal.png"),
            "visible": os.path.join(sdir, f"render_{scene_id:04d}_masks", f"obj_{obj_id:04d}_visible.png"),
            "meta": os.path.join(sdir, f"render_{scene_id:04d}_metadata.json"),
            "pose": os.path.join(self.pose_cache_dir, f"render_{scene_id:04d}.npz"),
        }

    def __getitem__(self, idx: int) -> dict[str, Any]:
        scene_id, obj_id = self.entries[idx]
        paths = self._paths(scene_id, obj_id)
        for key in ("img", "amodal", "visible", "meta", "pose"):
            if not os.path.exists(paths[key]):
                raise FileNotFoundError(f"missing {key} for scene={scene_id} obj={obj_id}: {paths[key]}")

        img_full = cv2.imread(paths["img"], cv2.IMREAD_GRAYSCALE)
        assert img_full is not None, f"cv2 failed to read {paths['img']}"
        img_full = img_full.astype(np.float32) / 255.0

        amodal_full = (cv2.imread(paths["amodal"], cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        visible_full = (cv2.imread(paths["visible"], cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)

        ys, xs = np.where(amodal_full > 0)
        if len(xs) == 0:
            raise RuntimeError(f"empty amodal for scene={scene_id} obj={obj_id}")
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1

        img_crop_raw, bbox_full, side_padded = _crop_square(
            img_full, (x0, y0, x1, y1), self.pad_ratio
        )
        amodal_crop_raw, _, _ = _crop_square(amodal_full, (x0, y0, x1, y1), self.pad_ratio)
        visible_crop_raw, _, _ = _crop_square(visible_full, (x0, y0, x1, y1), self.pad_ratio)

        # Resize to target
        img_crop = cv2.resize(img_crop_raw, (self.crop_size, self.crop_size), interpolation=cv2.INTER_LINEAR)
        amodal_crop = cv2.resize(
            amodal_crop_raw, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST
        )
        visible_crop = cv2.resize(
            visible_crop_raw, (self.crop_size, self.crop_size), interpolation=cv2.INTER_NEAREST
        )

        scale_pix = self.crop_size / side_padded  # full-pix → crop-pix per unit
        bx0, by0, bx1, by1 = bbox_full
        crop_affine = np.array(
            [
                [scale_pix, 0.0, -scale_pix * bx0],
                [0.0, scale_pix, -scale_pix * by0],
            ],
            dtype=np.float32,
        )

        # Load pose cache
        d = np.load(paths["pose"])
        obj_ids_cache = d["obj_ids"].tolist()
        i_obj = obj_ids_cache.index(int(obj_id))
        R = d["R"][i_obj].astype(np.float32)
        t = d["t"][i_obj].astype(np.float32)
        s_val = float(d["s"][i_obj])
        vertices_obj = d["vertices_obj"].astype(np.float32)  # (14, 3)
        vertices_cam = (s_val * vertices_obj @ R.T + t).astype(np.float32)

        kp_2d_full = vertices_cam[:, :2]  # orthographic
        kp_2d_crop = (kp_2d_full @ crop_affine[:, :2].T) + crop_affine[:, 2]

        # Visibility from metadata
        with open(paths["meta"], "r") as f:
            meta = json.load(f)
        visibility_val = 1.0
        for obj in meta:
            if int(obj["id"]) == int(obj_id):
                visibility_val = float(obj.get("visibility", 1.0))
                break

        img_tensor = torch.from_numpy(img_crop).unsqueeze(0).float() * 2.0 - 1.0
        amodal_tensor = torch.from_numpy(amodal_crop.astype(bool)).unsqueeze(0)
        visible_tensor = torch.from_numpy(visible_crop.astype(bool)).unsqueeze(0)

        return {
            "img_crop": img_tensor,
            "mask_visible": visible_tensor,
            "mask_amodal": amodal_tensor,
            "R": torch.from_numpy(R),
            "t": torch.from_numpy(t),
            "scale": torch.tensor(s_val, dtype=torch.float32),
            "K_ortho": torch.tensor(
                [scale_pix, scale_pix, -scale_pix * bx0, -scale_pix * by0],
                dtype=torch.float32,
            ),
            "bbox_full": torch.tensor([bx0, by0, bx1, by1], dtype=torch.float32),
            "crop_affine": torch.from_numpy(crop_affine),
            "vertices_obj": torch.from_numpy(vertices_obj),
            "vertices_cam": torch.from_numpy(vertices_cam),
            "kp_2d_crop": torch.from_numpy(kp_2d_crop.astype(np.float32)),
            "visibility": torch.tensor(visibility_val, dtype=torch.float32),
            "obj_id": torch.tensor(int(obj_id), dtype=torch.long),
            "scene_id": torch.tensor(int(scene_id), dtype=torch.long),
        }


def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
    """Default batch collation — stacks tensors along dim 0."""
    out: dict[str, Any] = {}
    for k in batch[0]:
        out[k] = torch.stack([b[k] for b in batch], dim=0)
    return out
