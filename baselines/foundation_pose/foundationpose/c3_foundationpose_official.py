"""C3 — FoundationPose (Wen et al., CVPR 2024) Case R+ official-checkpoint port.

Backbone is the pretrained `encodeA` CNN from FoundationPose's refiner network,
loaded from the paper's `weights/2023-10-28-18-33-37/model_best.pth` (refiner).
On top: small MLP pose head producing Option-D crop-norm (rot6d, t_norm, scale).

Why backbone-finetune (not register()-finetune): the paper's `register()` pipeline
is non-differentiable (252 SE(3) hypothesis sampling + scorer ranking). To match
the C1/C2 paradigm (paper backbone + Option-D head) we use only the encodeA module
as a feature extractor and train end-to-end with our standard losses.

Input has 6 channels: 3 ch RGB (grayscale replicated) + 3 ch xyz_map (per-pixel
world coords reconstructed from `crop_affine`, with the object plane at depth z_mean
inside the mask, zero outside). This matches FoundationPose's c_in=6 contract.
"""

from __future__ import annotations

import os
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.candidates.a1_gdrnet import rot6d_to_matrix, sym_pm_loss
from common.coord_transform import world_to_crop_t_torch, crop_to_world_t_torch
from common.geometry import O24_rotations


N_VERTICES = 14
FOUNDATIONPOSE_CKPT = (
    "<PROJECT_ROOT>/baselines/external/FoundationPose/weights/"
    "2023-10-28-18-33-37/model_best.pth"
)
Z_MEAN = 11.08  # Option-D scene-z prior
SCALE_MEAN = 40.58


# Mirror of FoundationPose/learning/models/network_modules.py — minimal copy
# (avoid sys.path injection and config import).
class _ConvBNReLU(nn.Module):
    def __init__(self, C_in, C_out, kernel_size=3, stride=1):
        super().__init__()
        padding = (kernel_size - 1) // 2
        self.net = nn.Sequential(
            nn.Conv2d(C_in, C_out, kernel_size, stride, padding, bias=True),
            nn.BatchNorm2d(C_out, momentum=0.1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class _ResnetBasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=True)
        self.bn1 = nn.BatchNorm2d(planes, momentum=0.1)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=True)
        self.bn2 = nn.BatchNorm2d(planes, momentum=0.1)
        self.relu = nn.ReLU(inplace=True)
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=True),
                nn.BatchNorm2d(planes, momentum=0.1),
            )
        else:
            self.downsample = None

    def forward(self, x):
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class _FoundationPoseEncodeA(nn.Module):
    """Mirror of FoundationPose RefineNet.encodeA — 4 blocks, c_in=6 → 128 ch."""

    def __init__(self, c_in: int = 6):
        super().__init__()
        self.net = nn.Sequential(
            _ConvBNReLU(c_in, 64, kernel_size=7, stride=2),
            _ConvBNReLU(64, 128, kernel_size=3, stride=2),
            _ResnetBasicBlock(128, 128),
            _ResnetBasicBlock(128, 128),
        )

    def forward(self, x):
        return self.net(x)


def _load_encodeA(load_paper_ckpt: bool = True) -> _FoundationPoseEncodeA:
    enc = _FoundationPoseEncodeA(c_in=6)
    if load_paper_ckpt and os.path.exists(FOUNDATIONPOSE_CKPT):
        ckpt = torch.load(FOUNDATIONPOSE_CKPT, map_location="cpu", weights_only=False)
        sd = ckpt["model"] if "model" in ckpt else ckpt
        # Keys like "encodeA.0.net.0.weight" → strip prefix to "0.net.0.weight"
        sub = {k[len("encodeA."):]: v for k, v in sd.items() if k.startswith("encodeA.")}
        # Our wrapping has an extra `.net` at the top, so prepend.
        remap = {f"net.{k}": v for k, v in sub.items()}
        missing, unexpected = enc.load_state_dict(remap, strict=False)
        print(f"[C3_foundationpose_official] loaded paper encodeA; "
              f"missing={len(missing)}, unexpected={len(unexpected)}",
              flush=True)
    return enc


def _build_xyz_map(crop_affine: torch.Tensor, mask: torch.Tensor,
                   crop_size: int) -> torch.Tensor:
    """Reconstruct per-pixel (x_world, y_world, z) inside the mask.
    Args:
        crop_affine: (B, 2, 3) — pix = A @ world_xy + b
        mask: (B, 1, H, W) bool — modal mask
        crop_size: H = W
    Returns: (B, 3, H, W) float32
    """
    B = crop_affine.shape[0]
    device = crop_affine.device
    H = W = crop_size
    A = crop_affine[:, :, :2]  # (B, 2, 2)
    b = crop_affine[:, :, 2]  # (B, 2)
    # pixel grid (W, H), broadcast (B, 2, H, W)
    ys, xs = torch.meshgrid(
        torch.arange(H, device=device, dtype=A.dtype),
        torch.arange(W, device=device, dtype=A.dtype),
        indexing="ij",
    )
    pix = torch.stack([xs, ys], dim=-1)  # (H, W, 2)
    pix = pix.unsqueeze(0).expand(B, -1, -1, -1)  # (B, H, W, 2)
    # world_xy = A^{-1} @ (pix - b)  per-batch
    A_inv = torch.linalg.inv(A)  # (B, 2, 2)
    delta = pix - b.view(B, 1, 1, 2)  # (B, H, W, 2)
    world_xy = torch.einsum("bij,bhwj->bhwi", A_inv, delta)  # (B, H, W, 2)
    z = torch.full((B, H, W, 1), Z_MEAN, device=device, dtype=A.dtype)
    xyz = torch.cat([world_xy, z], dim=-1)  # (B, H, W, 3)
    xyz = xyz.permute(0, 3, 1, 2)  # (B, 3, H, W)
    # Zero outside mask
    xyz = xyz * mask.to(xyz.dtype)
    return xyz


class C3FoundationPoseOfficialNet(nn.Module):
    def __init__(self, load_paper_ckpt: bool = True):
        super().__init__()
        self.encoder = _load_encodeA(load_paper_ckpt=load_paper_ckpt)
        # encodeA output: (B, 128, H/4, W/4). For 256 input → (B, 128, 64, 64).
        self.image_pool = nn.AdaptiveAvgPool2d(4)
        img_flat = 128 * 4 * 4
        self.image_proj = nn.Sequential(
            nn.Linear(img_flat, 1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.GELU(),
        )
        self.mesh_proj = nn.Sequential(
            nn.Linear(N_VERTICES * 3, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(1024 + 128, 512), nn.GELU(),
            nn.Linear(512, 6 + 3 + 1),
        )

    def forward(self, img, canonical_verts, mask, crop_affine):
        # img: (B, 1, H, W) in [-1, 1]; expand to RGB
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        rgb = img * 0.5 + 0.5  # → [0, 1]
        # xyz channels 3
        xyz = _build_xyz_map(crop_affine, mask, crop_size=img.shape[-1])
        # Normalize xyz to a similar scale as RGB (roughly [-1, 1] after dividing by ~world unit range)
        # World coords for our object are roughly in [0, 1024] → divide by ~512 then center.
        xyz_norm = xyz / 512.0
        x = torch.cat([rgb, xyz_norm], dim=1)  # (B, 6, H, W)

        feat = self.encoder(x)  # (B, 128, H/4, W/4)
        feat = self.image_pool(feat).flatten(1)
        feat = self.image_proj(feat)
        mesh_feat = self.mesh_proj(canonical_verts.flatten(1))
        out = self.head(torch.cat([feat, mesh_feat], dim=-1))
        rot6d = out[..., :6]
        t_norm = out[..., 6:9]
        s_norm = out[..., 9]
        R = rot6d_to_matrix(rot6d)
        return R, t_norm, s_norm


class C3FoundationPoseOfficialCandidate:
    name = "C3_foundationpose_official"
    category = "C"
    crop_size = 256
    train_batch_size = 8
    max_steps = 2000

    def __init__(self):
        self._syms = None

    def build_model(self, device, load_paper_ckpt: bool = True):
        m = C3FoundationPoseOfficialNet(load_paper_ckpt=load_paper_ckpt).to(device)
        self._syms = torch.from_numpy(O24_rotations()).to(device)
        return m

    def build_optimizer(self, model, max_steps):
        m = model.module if hasattr(model, "module") else model
        backbone_params = list(m.encoder.parameters())
        head_params = (list(m.image_proj.parameters())
                       + list(m.mesh_proj.parameters())
                       + list(m.head.parameters()))
        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": 5e-5},
            {"params": head_params, "lr": 2e-4},
        ], weight_decay=1e-4)
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=200)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_steps - 200), eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[200])
        return opt, sch

    def forward_train(self, model, batch):
        R, t_norm, s_norm = model(
            batch["img_crop"], batch["vertices_obj"],
            batch["mask_visible"], batch["crop_affine"],
        )
        t_norm_gt, s_norm_gt = world_to_crop_t_torch(
            batch["t"], batch["scale"], batch["crop_affine"], self.crop_size,
        )
        L_R = torch.norm(R - batch["R"], p="fro", dim=(-2, -1)).mean()
        L_t = F.smooth_l1_loss(t_norm, t_norm_gt)
        L_s = F.smooth_l1_loss(s_norm, s_norm_gt)
        t_world, s_world = crop_to_world_t_torch(
            t_norm, s_norm, batch["crop_affine"], self.crop_size,
        )
        L_PM = sym_pm_loss(
            R, t_world, s_world, batch["R"], batch["t"], batch["scale"],
            batch["vertices_obj"][0], self._syms,
        )
        loss = L_R + L_t + L_s + L_PM
        return {"loss": loss, "L_R": L_R.detach(), "L_t": L_t.detach(),
                "L_s": L_s.detach(), "L_PM": L_PM.detach()}

    def forward_eval(self, model_unwrapped, batch):
        with torch.no_grad():
            R, t_norm, s_norm = model_unwrapped(
                batch["img_crop"], batch["vertices_obj"],
                batch["mask_visible"], batch["crop_affine"],
            )
            t_world, s_world = crop_to_world_t_torch(
                t_norm, s_norm, batch["crop_affine"], self.crop_size,
            )
        return {"R_pred": R, "t_pred": t_world, "scale_pred": s_world}
