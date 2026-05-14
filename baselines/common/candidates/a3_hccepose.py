"""A3 — HccePose (Wang et al., RA-L 2024 / ICCV'25 Highlight) Case R+.

Wraps repo's `HccePose_BF_Net` (front + back HCCE codes, repo unmodified).
Adds a small tz-regression head to compensate for orthographic camera lacking
depth observation (documented deviation in code_audit.md).
"""

from __future__ import annotations

import os
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

REPO_ROOT = "<PROJECT_ROOT>/baselines/geometry_aware_pose/hccepose/repo"
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from HccePose.network_model import HccePose_BF_Net  # noqa: E402

from common.rasterize_back import hcce_decode_to_normalized
from common.coord_transform import load_stats

CODE_HW = 128
BITS_PER_DIM = 8
N_FRONT_BACK = 24


class _A3Wrapper(nn.Module):
    def __init__(self):
        super().__init__()
        self.hccepose = HccePose_BF_Net(efficientnet_key=None, input_channels=3)
        self.tz_backbone = nn.Sequential(
            nn.Conv2d(1, 32, 5, 2, 2), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 128, 3, 2, 1), nn.GELU(),
            nn.Conv2d(128, 128, 3, 2, 1), nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.tz_fc = nn.Linear(128, 1)

    def forward(self, img_1ch: torch.Tensor):
        img_3ch = img_1ch.expand(-1, 3, -1, -1)
        mask_logit, code_logit = self.hccepose(img_3ch)  # (B,1,128,128), (B,48,128,128)
        front = code_logit[:, :N_FRONT_BACK]
        back = code_logit[:, N_FRONT_BACK:N_FRONT_BACK * 2]
        tz_feat = self.tz_backbone(img_1ch).flatten(1)
        tz_norm = self.tz_fc(tz_feat).squeeze(-1)  # Option D: normalized scalar
        return mask_logit, front, back, tz_norm


def _load_gt(batch: dict, device: torch.device, cache_dir: str):
    scenes = batch["scene_id"].cpu().numpy()
    objs = batch["obj_id"].cpu().numpy()
    fronts, backs, masks = [], [], []
    legacy = "<PROJECT_ROOT>/baselines/phase0_A/hcce_code/overfit"
    for sid, oid in zip(scenes, objs):
        path = os.path.join(cache_dir, f"{int(sid):04d}_{int(oid):04d}.npz")
        if not os.path.exists(path):
            path = os.path.join(legacy, f"{int(sid):04d}_{int(oid):04d}.npz")
        d = np.load(path)
        fronts.append(d["front_code"])
        backs.append(d["back_code"])
        m = d["valid_front"].astype(np.float32)
        masks.append(m)
    front_t = torch.from_numpy(np.stack(fronts)).to(device)
    back_t = torch.from_numpy(np.stack(backs)).to(device)
    mask_t = torch.from_numpy(np.stack(masks)).unsqueeze(1).to(device)
    # resize to model output (128) — repo loss requires matching shapes
    if mask_t.shape[-1] != CODE_HW:
        mask_t = F.interpolate(mask_t, size=(CODE_HW, CODE_HW), mode="nearest")
        front_t = F.interpolate(front_t, size=(CODE_HW, CODE_HW), mode="nearest")
        back_t = F.interpolate(back_t, size=(CODE_HW, CODE_HW), mode="nearest")
    # repo's HccePose_Loss does pred_mask[:, 0] then expects gt_mask of (B, H, W)
    return mask_t.squeeze(1), front_t, back_t


def _decode_pose(
    front_logit: torch.Tensor, back_logit: torch.Tensor, mask_logit: torch.Tensor,
    tz_pred: torch.Tensor, batch: dict,
):
    """Per-pixel HCCE decode → canonical xyz → ortho factorization."""
    B = front_logit.shape[0]
    crop_affine = batch["crop_affine"].detach().cpu().numpy()

    front_p = torch.sigmoid(front_logit).detach().cpu().numpy()  # (B, 24, h, w)
    mask_p = torch.sigmoid(mask_logit).detach().cpu().numpy()[:, 0]  # (B, h, w)
    tz_np = tz_pred.detach().cpu().numpy()

    Rs, ts, ss = [], [], []
    for bi in range(B):
        m = mask_p[bi] > 0.5
        if int(m.sum()) < 50:
            Rs.append(np.eye(3, dtype=np.float32)); ts.append(np.zeros(3, dtype=np.float32)); ss.append(1.0); continue
        # binarize bits and decode normalized canonical xyz
        bits = (front_p[bi] > 0.5).astype(np.float32).transpose(1, 2, 0)  # (h, w, 24)
        norm = hcce_decode_to_normalized(bits, bits=BITS_PER_DIM)  # (h, w, 3) ∈ [0, 1]
        canonical = (norm * 4.0 - 2.0)  # back to canonical [-2, 2]
        h, w = norm.shape[:2]

        # collect (canonical_xyz, crop_pixel) correspondences from foreground
        ys, xs = np.where(m)
        if len(xs) < 50:
            Rs.append(np.eye(3, dtype=np.float32)); ts.append(np.zeros(3, dtype=np.float32)); ss.append(1.0); continue
        # subsample to keep linear solve fast
        max_pts = 1500
        if len(xs) > max_pts:
            idx = np.random.RandomState(0).choice(len(xs), size=max_pts, replace=False)
            xs = xs[idx]; ys = ys[idx]
        canonical_pts = canonical[ys, xs]  # (N, 3)
        # model output is 128, crop_affine maps world→256-pixel; scale by 2
        cx_256 = 2.0 * xs.astype(np.float32)
        cy_256 = 2.0 * ys.astype(np.float32)
        A_mat = crop_affine[bi, :, :2]
        t_aff = crop_affine[bi, :, 2]
        # invert: world = solve(A, pixel - t)
        pix = np.stack([cx_256, cy_256], axis=1) - t_aff[None]
        world_xy = np.linalg.solve(A_mat[None].repeat(len(xs), axis=0), pix[..., None]).squeeze(-1)
        # actually solve once per row, easier:
        A_inv = np.linalg.inv(A_mat)
        world_xy = pix @ A_inv.T  # (N, 2)

        # least squares: [c_x c_y c_z 1] @ [a tx]^T = wx; same for wy
        X = np.hstack([canonical_pts, np.ones((len(canonical_pts), 1))])  # (N, 4)
        coef_x, *_ = np.linalg.lstsq(X, world_xy[:, 0], rcond=None)
        coef_y, *_ = np.linalg.lstsq(X, world_xy[:, 1], rcond=None)
        a = coef_x[:3]; b_vec = coef_y[:3]
        tx = float(coef_x[3]); ty = float(coef_y[3])
        s_a = np.linalg.norm(a); s_b = np.linalg.norm(b_vec)
        s_val = 0.5 * (s_a + s_b)
        if s_val < 1e-6:
            Rs.append(np.eye(3, dtype=np.float32)); ts.append(np.zeros(3, dtype=np.float32)); ss.append(1.0); continue
        r0 = a / s_a; r1 = b_vec / s_b
        r1 = r1 - np.dot(r1, r0) * r0
        nrm = np.linalg.norm(r1)
        if nrm < 1e-6:
            Rs.append(np.eye(3, dtype=np.float32)); ts.append(np.zeros(3, dtype=np.float32)); ss.append(1.0); continue
        r1 = r1 / nrm
        r2 = np.cross(r0, r1)
        R = np.stack([r0, r1, r2], axis=0).astype(np.float32)
        if np.linalg.det(R) < 0:
            r2 = -r2
            R = np.stack([r0, r1, r2], axis=0).astype(np.float32)
        t = np.array([tx, ty, float(tz_np[bi])], dtype=np.float32)
        Rs.append(R); ts.append(t); ss.append(np.float32(s_val))

    return {
        "R_pred": torch.from_numpy(np.stack(Rs)).to(front_logit.device),
        "t_pred": torch.from_numpy(np.stack(ts)).to(front_logit.device),
        "scale_pred": torch.from_numpy(np.asarray(ss, dtype=np.float32)).to(front_logit.device),
    }


class A3HcceposeCandidate:
    name = "A3_hccepose"
    category = "A"
    crop_size = 256
    train_batch_size = 4
    max_steps = 2000
    # Unified HCCE cache covers train/val/test/overfit/val_viz.
    # Falls back to legacy overfit dir if unified cache is missing.
    cache_dir = "<PROJECT_ROOT>/baselines/phase0_A/hcce_code/all"

    def __init__(self):
        pass

    def build_model(self, device):
        model = _A3Wrapper().to(device)
        return model

    def build_optimizer(self, model, max_steps):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=200)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_steps - 200), eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[200])
        return opt, sch

    def forward_train(self, model, batch):
        """Simplified HccePose loss (no dynamic weighting; repo class has a
        numpy↔cuda assignment bug under torch 2.x strict typing). Functional
        form (mask BCE + per-bit BCE on front/back codes inside fg mask) follows
        paper Sec 3.4. Documented as Case R+ deviation.
        """
        device = batch["img_crop"].device
        mask_logit, front_logit, back_logit, tz_norm_pred = model(batch["img_crop"])
        mask_gt, front_gt, back_gt = _load_gt(batch, device, self.cache_dir)
        mask_gt_4d = mask_gt.unsqueeze(1)
        L_mask = F.binary_cross_entropy_with_logits(mask_logit, mask_gt_4d)
        L_front = F.binary_cross_entropy_with_logits(front_logit, front_gt, reduction="none")
        L_front = (L_front * mask_gt_4d).sum() / (mask_gt_4d.sum().clamp(min=1) * 24)
        L_back = F.binary_cross_entropy_with_logits(back_logit, back_gt, reduction="none")
        L_back = (L_back * mask_gt_4d).sum() / (mask_gt_4d.sum().clamp(min=1) * 24)
        # Option D: tz target normalized
        stats = load_stats()
        tz_norm_gt = (batch["t"][:, 2] - stats["tz_mean"]) / stats["tz_std"]
        L_tz = F.smooth_l1_loss(tz_norm_pred, tz_norm_gt)
        loss = 10.0 * L_mask + L_front + L_back + L_tz
        return {
            "loss": loss,
            "L_mask": L_mask.detach(),
            "L_front": L_front.detach(),
            "L_back": L_back.detach(),
            "L_tz": L_tz.detach(),
        }

    def forward_eval(self, model_unwrapped, batch):
        with torch.no_grad():
            mask_logit, front_logit, back_logit, tz_norm_pred = model_unwrapped(batch["img_crop"])
            stats = load_stats()
            tz_pred_world = tz_norm_pred * stats["tz_std"] + stats["tz_mean"]
        return _decode_pose(front_logit, back_logit, mask_logit, tz_pred_world, batch)
