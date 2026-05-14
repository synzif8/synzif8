"""B3 — Uni6D (Sun et al., CVPR 2022) Case R- RGB-only port.

ResNet-50 backbone + global pool + MLP → 14 × 3 camera-frame coords.
Architectural distinction from B1 REDE (ResNet-34): deeper backbone.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50

from common.candidates.b1_rede import sym_mpjpe_loss
from common.coord_transform import world_to_crop_kp3d_torch, crop_to_world_kp3d_torch
from common.geometry import O24_rotations


N_VERTICES = 14


class _ResNet50Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        net = resnet50(weights=None)
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
        x = self.layer4(x)  # (B, 2048, H/32, W/32)
        return x


class B3Uni6DNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.backbone = _ResNet50Backbone()
        self.pool = nn.AdaptiveAvgPool2d(8)
        flat = 2048 * 8 * 8
        self.mlp = nn.Sequential(
            nn.Linear(flat, 1024), nn.GELU(),
            nn.Linear(1024, 1024), nn.GELU(),
            nn.Linear(1024, N_VERTICES * 3),
        )

    def forward(self, img):
        if img.shape[1] == 1:
            img = img.expand(-1, 3, -1, -1)
        feat = self.backbone(img)
        x = self.pool(feat).flatten(1)
        out = self.mlp(x)
        return out.view(-1, N_VERTICES, 3)


class B3Uni6DCandidate:
    name = "B3_uni6d"
    category = "B"
    crop_size = 256
    train_batch_size = 4
    max_steps = 2000

    def __init__(self):
        self._syms = None

    def build_model(self, device):
        m = B3Uni6DNet().to(device)
        self._syms = torch.from_numpy(O24_rotations()).to(device)
        return m

    def build_optimizer(self, model, max_steps):
        opt = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=1e-4)
        warm = torch.optim.lr_scheduler.LinearLR(opt, start_factor=0.1, total_iters=200)
        cos = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(1, max_steps - 200), eta_min=1e-6)
        sch = torch.optim.lr_scheduler.SequentialLR(opt, schedulers=[warm, cos], milestones=[200])
        return opt, sch

    def forward_train(self, model, batch):
        kp_norm = model(batch["img_crop"])
        kp_norm_gt = world_to_crop_kp3d_torch(
            batch["vertices_cam"], batch["crop_affine"], self.crop_size,
        )
        L_direct = F.smooth_l1_loss(kp_norm, kp_norm_gt)
        kp_world = crop_to_world_kp3d_torch(kp_norm, batch["crop_affine"], self.crop_size)
        L_sym = sym_mpjpe_loss(
            kp_world, batch["R"], batch["t"], batch["scale"],
            batch["vertices_obj"][0], self._syms,
        )
        loss = L_sym + L_direct
        return {"loss": loss, "L_sym": L_sym.detach(), "L_direct": L_direct.detach()}

    def forward_eval(self, model_unwrapped, batch):
        with torch.no_grad():
            kp_norm = model_unwrapped(batch["img_crop"])
            kp_world = crop_to_world_kp3d_torch(kp_norm, batch["crop_affine"], self.crop_size)
        return {"kp3d_pred": kp_world}
