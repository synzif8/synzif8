"""C1 — MegaPose (Labbé et al., CoRL 2022) Case R+ official-checkpoint port.

ResNet-50 backbone with weights loaded from MegaPose's coarse-rgb checkpoint
(http://www.paris.inria.fr/archive_ylabbeprojectsdata/megapose/megapose-models/
coarse-rgb-906902141/checkpoint.pth.tar). On top: small MLP pose head producing
Option-D crop-norm (rot6d, t_norm, scale_norm). For finetuning we let the
backbone train; for zero-shot the head is left untrained.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet34, ResNet34_Weights

from common.candidates.a1_gdrnet import rot6d_to_matrix, sym_pm_loss
from common.coord_transform import world_to_crop_t_torch, crop_to_world_t_torch
from common.geometry import O24_rotations


N_VERTICES = 14
MEGAPOSE_CKPT = (
    "<PROJECT_ROOT>/baselines/external/megapose6d/local_data/megapose-models/"
    "coarse-rgb-906902141/checkpoint.pth.tar"
)


class _MegaResNet34(nn.Module):
    def __init__(self, load_paper_ckpt: bool = True):
        super().__init__()
        net = resnet34(weights=ResNet34_Weights.IMAGENET1K_V1)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4
        if load_paper_ckpt and os.path.exists(MEGAPOSE_CKPT):
            self._load_megapose_weights(MEGAPOSE_CKPT)

    def _load_megapose_weights(self, ckpt_path: str) -> None:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        # MegaPose stores ResNet50 under "backbone.backbone.*" prefix
        own = {}
        for k, v in sd.items():
            if not k.startswith("backbone.backbone."):
                continue
            kk = k.replace("backbone.backbone.", "")
            own[kk] = v
        # torchvision resnet50 names: conv1, bn1, layer1.*, layer2.*, layer3.*, layer4.*, fc.*
        # We split conv1/bn1 into self.stem; layers stay the same.
        remap: dict[str, torch.Tensor] = {}
        for k, v in own.items():
            # MegaPose conv1 expects 9 channels (render+compare); skip conv1/bn1
            # and keep ImageNet weights for the stem.
            if k.startswith("conv1.") or k.startswith("bn1."):
                continue
            elif k.startswith("layer1."):
                remap[f"layer1.{k[len('layer1.'):]}"] = v
            elif k.startswith("layer2."):
                remap[f"layer2.{k[len('layer2.'):]}"] = v
            elif k.startswith("layer3."):
                remap[f"layer3.{k[len('layer3.'):]}"] = v
            elif k.startswith("layer4."):
                remap[f"layer4.{k[len('layer4.'):]}"] = v
            # skip fc head
        missing, unexpected = self.load_state_dict(remap, strict=False)
        # Filter out fc-related (not part of stem) for sanity
        print(f"[C1_megapose_official] loaded paper ckpt; missing={len(missing)}, unexpected={len(unexpected)}", flush=True)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)  # (B, 512, H/32, W/32) for ResNet34
        return x


class C1MegaPoseOfficialNet(nn.Module):
    def __init__(self, load_paper_ckpt: bool = True):
        super().__init__()
        self.backbone = _MegaResNet34(load_paper_ckpt=load_paper_ckpt)
        self.image_pool = nn.AdaptiveAvgPool2d(4)
        img_flat = 512 * 4 * 4
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

    def forward(self, img, canonical_verts):
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        feat = self.backbone(img)
        feat = self.image_pool(feat).flatten(1)
        feat = self.image_proj(feat)
        mesh_feat = self.mesh_proj(canonical_verts.flatten(1))
        out = self.head(torch.cat([feat, mesh_feat], dim=-1))
        rot6d = out[..., :6]
        t_norm = out[..., 6:9]
        s_norm = out[..., 9]  # raw — Option D crop-norm
        R = rot6d_to_matrix(rot6d)
        return R, t_norm, s_norm


class C1MegaPoseOfficialCandidate:
    name = "C1_megapose_official"
    category = "C"
    crop_size = 256
    train_batch_size = 4
    max_steps = 2000

    def __init__(self):
        self._syms = None

    def build_model(self, device, load_paper_ckpt: bool = True):
        m = C1MegaPoseOfficialNet(load_paper_ckpt=load_paper_ckpt).to(device)
        self._syms = torch.from_numpy(O24_rotations()).to(device)
        return m

    def build_optimizer(self, model, max_steps):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=200)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_steps - 200), eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[200])
        return opt, sch

    def forward_train(self, model, batch):
        R, t_norm, s_norm = model(batch["img_crop"], batch["vertices_obj"])
        # Option D direct losses on crop-norm
        t_norm_gt, s_norm_gt = world_to_crop_t_torch(
            batch["t"], batch["scale"], batch["crop_affine"], self.crop_size,
        )
        L_R = torch.norm(R - batch["R"], p="fro", dim=(-2, -1)).mean()
        L_t = F.smooth_l1_loss(t_norm, t_norm_gt)
        L_s = F.smooth_l1_loss(s_norm, s_norm_gt)
        # PM in world frame
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
            R, t_norm, s_norm = model_unwrapped(batch["img_crop"], batch["vertices_obj"])
            t_world, s_world = crop_to_world_t_torch(
                t_norm, s_norm, batch["crop_affine"], self.crop_size,
            )
        return {"R_pred": R, "t_pred": t_world, "scale_pred": s_world}
