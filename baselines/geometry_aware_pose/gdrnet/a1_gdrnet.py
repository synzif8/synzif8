"""A1 — GDR-Net (Wang et al., CVPR 2021), Case R- paper-faithful reimplementation.

See runs/A_appearance/A1_gdrnet/code_audit.md for deviation log.

Direct-variant (Fig. 2 bottom): geometry head (mask + NOCS) + MLP pose head
(6D rot + trans + scale). Region head omitted (Patch-PnP not used).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34

from common.geometry import O24_rotations
from common.rasterize_nocs import render_nocs_crop
from common.coord_transform import world_to_crop_t_torch, crop_to_world_t_torch


def rot6d_to_matrix(rot6d: torch.Tensor) -> torch.Tensor:
    """Zhou et al. 2019 — map 6D to SO(3) via Gram-Schmidt."""
    a1 = rot6d[..., 0:3]
    a2 = rot6d[..., 3:6]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack([b1, b2, b3], dim=-1)  # (..., 3, 3)


class GeometryHead(nn.Module):
    """Upsample 8× feature → 64×64 mask + NOCS maps."""

    def __init__(self, in_ch: int = 512, mid_ch: int = 256):
        super().__init__()

        def deconv(ic, oc):
            return nn.Sequential(
                nn.ConvTranspose2d(ic, oc, 3, stride=2, padding=1, output_padding=1, bias=False),
                nn.BatchNorm2d(oc),
                nn.GELU(),
            )

        self.up1 = deconv(in_ch, mid_ch)
        self.up2 = deconv(mid_ch, mid_ch)
        self.up3 = deconv(mid_ch, mid_ch)
        self.mask_head = nn.Conv2d(mid_ch, 1, kernel_size=1)
        self.nocs_head = nn.Conv2d(mid_ch, 3, kernel_size=1)

    def forward(self, feat):
        x = self.up1(feat)
        x = self.up2(x)
        x = self.up3(x)
        mask = self.mask_head(x)
        nocs = torch.tanh(self.nocs_head(x))  # [-1, 1]
        return mask, nocs


class PoseHead(nn.Module):
    """Flatten + MLP → (R_6d, t_xyz, s)."""

    def __init__(self, in_ch: int = 512, pool_hw: int = 8, hidden: int = 512):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(pool_hw)
        flat_dim = in_ch * pool_hw * pool_hw
        self.mlp = nn.Sequential(
            nn.Linear(flat_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, 6 + 3 + 1),
        )

    def forward(self, feat):
        x = self.pool(feat).flatten(1)
        out = self.mlp(x)
        rot6d = out[..., :6]
        trans = out[..., 6:9]   # crop-norm (Option D)
        scale = out[..., 9]      # crop-norm (Option D), raw — no softplus
        return rot6d, trans, scale


class _ResNet34Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet34(weights=None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # 8× downsampled, 512ch
        return x


class GDRNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _ResNet34Backbone()
        self.geo_head = GeometryHead(in_ch=512, mid_ch=256)
        self.pose_head = PoseHead(in_ch=512, pool_hw=8, hidden=512)

    def forward(self, img: torch.Tensor) -> dict:
        # img: (B, 1, H, W) grayscale → replicate to 3 channels
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        feat = self.backbone(img)  # (B, 512, H/32, W/32)
        mask_logit, nocs_map = self.geo_head(feat)
        rot6d, trans, scale = self.pose_head(feat)
        R = rot6d_to_matrix(rot6d)
        return {
            "mask_logit": mask_logit,
            "nocs": nocs_map,
            "R": R,
            "t": trans,
            "scale": scale,
        }


def _resize_nocs_gt(nocs_hw3: torch.Tensor, valid_hw: torch.Tensor, target_hw: int) -> tuple[torch.Tensor, torch.Tensor]:
    """(B,H,W,3), (B,H,W) → (B,3,target,target), (B,1,target,target) mask."""
    B = nocs_hw3.shape[0]
    nocs = nocs_hw3.permute(0, 3, 1, 2)  # (B, 3, H, W)
    valid = valid_hw.unsqueeze(1).float()  # (B, 1, H, W)
    nocs_r = F.interpolate(nocs, size=(target_hw, target_hw), mode="bilinear", align_corners=False)
    valid_r = F.interpolate(valid, size=(target_hw, target_hw), mode="nearest")
    return nocs_r, valid_r


def sym_pm_loss(
    R_pred: torch.Tensor, t_pred: torch.Tensor, s_pred: torch.Tensor,
    R_gt: torch.Tensor, t_gt: torch.Tensor, s_gt: torch.Tensor,
    canonical: torch.Tensor, syms: torch.Tensor,
) -> torch.Tensor:
    """Symmetry-aware Point Matching loss (paper Eq 6) averaged over batch.

    canonical: (V, 3), syms: (S, 3, 3).
    """
    B = R_pred.shape[0]
    V = canonical.shape[0]
    # pred vertices: (B, V, 3)
    pred_v = torch.einsum("bij,vj->bvi", R_pred, canonical) * s_pred.view(B, 1, 1) + t_pred.view(B, 1, 3)
    # gt vertices under each sym G: (B, S, V, 3)
    canon_sym = torch.einsum("sij,vj->svi", syms, canonical)  # (S, V, 3)
    gt_sym = torch.einsum("bij,svj->bsvi", R_gt, canon_sym) * s_gt.view(B, 1, 1, 1) + t_gt.view(B, 1, 1, 3)
    # distance per symmetry: (B, S)
    dists = (pred_v.unsqueeze(1) - gt_sym).norm(dim=-1).mean(dim=-1)
    return dists.min(dim=1).values.mean()


class A1GdrnetCandidate:
    name = "A1_gdrnet"
    category = "A"
    crop_size = 256
    train_batch_size = 4
    max_steps = 2000
    geo_hw = 64  # NOCS head resolution

    def __init__(self):
        self._syms = None
        self._canon = None

    def build_model(self, device):
        model = GDRNet().to(device)
        self._syms = torch.from_numpy(O24_rotations()).to(device)  # (24, 3, 3)
        return model

    def build_optimizer(self, model, max_steps):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=200)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_steps - 200), eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[200])
        return opt, sch

    def _ensure_nocs_gt(self, batch, device):
        """Compute NOCS GT on the fly from canonical + pose + crop_affine.

        Using the rasterizer is CPU. For 4-GPU x batch=4, a handful of calls per step
        on 256×256 crops. Acceptable for overfit-1.
        """
        B = batch["img_crop"].shape[0]
        canon = batch["vertices_obj"].cpu().numpy()
        vcam = batch["vertices_cam"].cpu().numpy()
        aff = batch["crop_affine"].cpu().numpy()
        H = batch["img_crop"].shape[-1]
        nocs_list, valid_list = [], []
        for i in range(B):
            nocs, valid = render_nocs_crop(canon[i], vcam[i], aff[i], crop_size=H)
            nocs_list.append(nocs)
            valid_list.append(valid)
        import numpy as np
        nocs_np = np.stack(nocs_list).astype(np.float32)
        valid_np = np.stack(valid_list)
        return (
            torch.from_numpy(nocs_np).to(device),
            torch.from_numpy(valid_np).to(device),
        )

    def forward_train(self, model, batch):
        device = batch["img_crop"].device
        out = model(batch["img_crop"])
        nocs_gt_hw3, valid_gt_hw = self._ensure_nocs_gt(batch, device)
        nocs_gt_r, mask_gt_r = _resize_nocs_gt(nocs_gt_hw3, valid_gt_hw, self.geo_hw)

        mask_logit = out["mask_logit"]
        nocs_pred = out["nocs"]

        L_mask = F.binary_cross_entropy_with_logits(mask_logit, mask_gt_r)
        L_nocs = (torch.abs(nocs_pred - nocs_gt_r) * mask_gt_r).sum() / (mask_gt_r.sum().clamp(min=1.0) * 3)
        L_R = torch.norm(out["R"] - batch["R"], p="fro", dim=(-2, -1)).mean()

        # Option D: normalized-space targets for t/scale
        t_norm_gt, s_norm_gt = world_to_crop_t_torch(
            batch["t"], batch["scale"], batch["crop_affine"], self.crop_size,
        )
        L_t = F.smooth_l1_loss(out["t"], t_norm_gt)
        L_s = F.smooth_l1_loss(out["scale"], s_norm_gt)

        # PM loss in world frame: unnormalize model output then sym-match
        t_world, s_world = crop_to_world_t_torch(
            out["t"], out["scale"], batch["crop_affine"], self.crop_size,
        )
        L_PM = sym_pm_loss(
            out["R"], t_world, s_world,
            batch["R"], batch["t"], batch["scale"],
            batch["vertices_obj"][0], self._syms,
        )
        loss = L_mask + L_nocs + L_R + L_t + L_s + L_PM

        return {
            "loss": loss,
            "L_mask": L_mask.detach(),
            "L_nocs": L_nocs.detach(),
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
