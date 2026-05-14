"""C2 — GigaPose (Nguyen et al., CVPR 2024) Case R+ official-checkpoint port.

DINOv2 ViT-L backbone with weights loaded from GigaPose's gigaPose_v1.ckpt
(huggingface.co/datasets/nv-nguyen/gigaPose). On top: small MLP pose head
producing Option-D crop-norm (rot6d, t_norm, scale_norm). For finetuning we
let the backbone train (or freeze it via env var); for zero-shot the head is
left untrained.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from common.candidates.a1_gdrnet import rot6d_to_matrix, sym_pm_loss
from common.coord_transform import world_to_crop_t_torch, crop_to_world_t_torch
from common.geometry import O24_rotations


N_VERTICES = 14
GIGAPOSE_CKPT = (
    "<PROJECT_ROOT>/baselines/external/gigapose/pretrained/gigaPose_v1.ckpt"
)
DINO_INPUT = 224  # multiple of 14
DINO_DIM = 1024  # vitl14


def _load_dinov2_vitl(load_paper_ckpt: bool = True) -> nn.Module:
    # Force fresh init (no pretrained), then overwrite from gigapose ckpt.
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitl14", pretrained=False)
    if load_paper_ckpt and os.path.exists(GIGAPOSE_CKPT):
        ckpt = torch.load(GIGAPOSE_CKPT, map_location="cpu", weights_only=False)
        sd = ckpt["state_dict"]
        sub = {k.replace("ae_net.dinov2_model.", ""): v
               for k, v in sd.items()
               if k.startswith("ae_net.dinov2_model.")}
        missing, unexpected = model.load_state_dict(sub, strict=False)
        print(f"[C2_gigapose_official] loaded gigapose DINOv2; missing={len(missing)}, unexpected={len(unexpected)}", flush=True)
    # Freeze params that aren't used in `forward_features` (avoids DDP unused-param error).
    if hasattr(model, "mask_token"):
        model.mask_token.requires_grad_(False)
    return model


class C2GigaPoseOfficialNet(nn.Module):
    def __init__(self, load_paper_ckpt: bool = True):
        super().__init__()
        self.dinov2 = _load_dinov2_vitl(load_paper_ckpt=load_paper_ckpt)
        # bf16 for xformers compatibility on Blackwell (SM 12.0)
        self.dinov2 = self.dinov2.to(torch.bfloat16)
        self.mesh_proj = nn.Sequential(
            nn.Linear(N_VERTICES * 3, 128), nn.GELU(),
            nn.Linear(128, 128), nn.GELU(),
        )
        self.head = nn.Sequential(
            nn.Linear(DINO_DIM + 128, 512), nn.GELU(),
            nn.Linear(512, 6 + 3 + 1),
        )
        # ImageNet normalization for DINOv2
        self.register_buffer("_mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("_std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, img, canonical_verts):
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        # img is in [-1, 1]; convert to ImageNet-normalized
        img = img * 0.5 + 0.5  # [0, 1]
        img = (img - self._mean) / self._std
        if img.shape[-1] != DINO_INPUT:
            img = F.interpolate(img, size=(DINO_INPUT, DINO_INPUT), mode="bilinear", align_corners=False)
        img_bf16 = img.to(torch.bfloat16)
        # CLS token from forward_features
        feats = self.dinov2.forward_features(img_bf16)
        cls = feats["x_norm_clstoken"].to(img.dtype)  # (B, 1024)
        mesh_feat = self.mesh_proj(canonical_verts.flatten(1))
        out = self.head(torch.cat([cls, mesh_feat], dim=-1))
        rot6d = out[..., :6]
        t_norm = out[..., 6:9]
        s_norm = out[..., 9]
        R = rot6d_to_matrix(rot6d)
        return R, t_norm, s_norm


class C2GigaPoseOfficialCandidate:
    name = "C2_gigapose_official"
    category = "C"
    crop_size = 256
    train_batch_size = 4
    max_steps = 2000

    def __init__(self):
        self._syms = None

    def build_model(self, device, load_paper_ckpt: bool = True):
        m = C2GigaPoseOfficialNet(load_paper_ckpt=load_paper_ckpt).to(device)
        self._syms = torch.from_numpy(O24_rotations()).to(device)
        return m

    def build_optimizer(self, model, max_steps):
        # Unwrap DDP if needed
        m = model.module if hasattr(model, "module") else model
        # Lower LR for the (huge) DINOv2 backbone to avoid catastrophic forgetting
        backbone_params = list(m.dinov2.parameters())
        head_params = list(m.mesh_proj.parameters()) + list(m.head.parameters())
        opt = torch.optim.AdamW([
            {"params": backbone_params, "lr": 1e-5},
            {"params": head_params, "lr": 2e-4},
        ], weight_decay=1e-4)
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=200)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_steps - 200), eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[200])
        return opt, sch

    def forward_train(self, model, batch):
        R, t_norm, s_norm = model(batch["img_crop"], batch["vertices_obj"])
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
            R, t_norm, s_norm = model_unwrapped(batch["img_crop"], batch["vertices_obj"])
            t_world, s_world = crop_to_world_t_torch(
                t_norm, s_norm, batch["crop_affine"], self.crop_size,
            )
        return {"R_pred": R, "t_pred": t_world, "scale_pred": s_world}
