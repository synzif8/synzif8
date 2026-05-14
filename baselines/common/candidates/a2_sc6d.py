"""A2 — SC6D (Cai et al., 3DV 2022), Case R- paper-faithful reimplementation.

See runs/A_appearance/A2_sc6d/code_audit.md for deviations.

SC6D is the "no dense correspondence" sibling of GDR-Net: pose head only, with
symmetry-agnostic point-matching supervision. We reuse our A1 backbone +
PoseHead and drop the geometry head entirely.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.candidates.a1_gdrnet import (
    _ResNet34Backbone, PoseHead, rot6d_to_matrix, sym_pm_loss,
)
from common.coord_transform import world_to_crop_t_torch, crop_to_world_t_torch
from common.geometry import O24_rotations


class SC6DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _ResNet34Backbone()
        self.pose_head = PoseHead(in_ch=512, pool_hw=8, hidden=512)

    def forward(self, img: torch.Tensor) -> dict:
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        feat = self.backbone(img)
        rot6d, trans, scale = self.pose_head(feat)
        R = rot6d_to_matrix(rot6d)
        return {"R": R, "t": trans, "scale": scale}


class A2Sc6dCandidate:
    name = "A2_sc6d"
    category = "A"
    crop_size = 256
    train_batch_size = 4
    max_steps = 2000

    def __init__(self):
        self._syms = None

    def build_model(self, device):
        m = SC6DNet().to(device)
        self._syms = torch.from_numpy(O24_rotations()).to(device)
        return m

    def build_optimizer(self, model, max_steps):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=200)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_steps - 200), eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[200])
        return opt, sch

    def forward_train(self, model, batch):
        out = model(batch["img_crop"])
        L_R = torch.norm(out["R"] - batch["R"], p="fro", dim=(-2, -1)).mean()
        # Option D: normalized-space targets
        t_norm_gt, s_norm_gt = world_to_crop_t_torch(
            batch["t"], batch["scale"], batch["crop_affine"], self.crop_size,
        )
        L_t = F.smooth_l1_loss(out["t"], t_norm_gt)
        L_s = F.smooth_l1_loss(out["scale"], s_norm_gt)
        # PM in world frame
        t_world, s_world = crop_to_world_t_torch(
            out["t"], out["scale"], batch["crop_affine"], self.crop_size,
        )
        L_PM = sym_pm_loss(
            out["R"], t_world, s_world,
            batch["R"], batch["t"], batch["scale"],
            batch["vertices_obj"][0], self._syms,
        )
        loss = L_R + L_t + L_s + L_PM
        return {
            "loss": loss,
            "L_R": L_R.detach(),
            "L_t": L_t.detach(),
            "L_s": L_s.detach(),
            "L_PM": L_PM.detach(),
        }

    def forward_eval(self, model_unwrapped, batch):
        with torch.no_grad():
            out = model_unwrapped(batch["img_crop"])
            t_world, s_world = crop_to_world_t_torch(
                out["t"], out["scale"], batch["crop_affine"], self.crop_size,
            )
        return {"R_pred": out["R"], "t_pred": t_world, "scale_pred": s_world}
