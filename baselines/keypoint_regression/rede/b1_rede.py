"""B1 — REDE (Hua et al., RA-L 2021) Case R- RGB-only port.

Direct 3D keypoint regression (camera frame). Pose decode via Umeyama.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from common.candidates.a1_gdrnet import _ResNet34Backbone
from common.coord_transform import world_to_crop_kp3d_torch, crop_to_world_kp3d_torch
from common.geometry import O24_rotations, umeyama_np


N_VERTICES = 14


class _RedeHead(nn.Module):
    def __init__(self, in_ch=512, pool_hw=8, hidden=512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_hw)
        flat = in_ch * pool_hw * pool_hw
        self.mlp = nn.Sequential(
            nn.Linear(flat, hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.GELU(),
            nn.Linear(hidden, N_VERTICES * 3),
        )

    def forward(self, feat):
        x = self.pool(feat).flatten(1)
        out = self.mlp(x)
        return out.view(-1, N_VERTICES, 3)


class B1RedeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _ResNet34Backbone()
        self.head = _RedeHead()

    def forward(self, img):
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        feat = self.backbone(img)
        return self.head(feat)  # (B, 14, 3)


def sym_mpjpe_loss(
    pred_kp: torch.Tensor,    # (B, V, 3)
    R_gt: torch.Tensor, t_gt: torch.Tensor, s_gt: torch.Tensor,
    canonical: torch.Tensor,  # (V, 3)
    syms: torch.Tensor,       # (S, 3, 3)
) -> torch.Tensor:
    B = pred_kp.shape[0]
    canon_sym = torch.einsum("sij,vj->svi", syms, canonical)  # (S, V, 3)
    gt_sym = torch.einsum("bij,svj->bsvi", R_gt, canon_sym) * s_gt.view(B, 1, 1, 1) + t_gt.view(B, 1, 1, 3)
    dists = (pred_kp.unsqueeze(1) - gt_sym).norm(dim=-1).mean(dim=-1)  # (B, S)
    return dists.min(dim=1).values.mean()


class B1RedeCandidate:
    name = "B1_rede"
    category = "B"
    crop_size = 256
    train_batch_size = 4
    max_steps = 2000

    def __init__(self):
        self._syms = None

    def build_model(self, device):
        m = B1RedeNet().to(device)
        self._syms = torch.from_numpy(O24_rotations()).to(device)
        return m

    def build_optimizer(self, model, max_steps):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=200)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_steps - 200), eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[200])
        return opt, sch

    def forward_train(self, model, batch):
        kp_norm_pred = model(batch["img_crop"])  # (B, 14, 3) crop-norm
        # Option D: normalize GT vertices to crop-norm space
        kp_norm_gt = world_to_crop_kp3d_torch(
            batch["vertices_cam"], batch["crop_affine"], self.crop_size,
        )
        L_direct = F.smooth_l1_loss(kp_norm_pred, kp_norm_gt)
        # Sym MPJPE in world frame: unnormalize → world keypoints, then sym match
        kp_world_pred = crop_to_world_kp3d_torch(
            kp_norm_pred, batch["crop_affine"], self.crop_size,
        )
        L_sym = sym_mpjpe_loss(
            kp_world_pred, batch["R"], batch["t"], batch["scale"],
            batch["vertices_obj"][0], self._syms,
        )
        loss = L_sym + L_direct
        return {"loss": loss, "L_sym": L_sym.detach(), "L_direct": L_direct.detach()}

    def forward_eval(self, model_unwrapped, batch):
        with torch.no_grad():
            kp_norm = model_unwrapped(batch["img_crop"])
            kp_world = crop_to_world_kp3d_torch(kp_norm, batch["crop_affine"], self.crop_size)
        return {"kp3d_pred": kp_world}
