# B3 Uni6D (RGB-only port) — Code Audit (Case R-)

## Repo provenance
- Source paper: Sun et al., "Uni6D: A Unified CNN Framework without Projection
  Breakdown for 6D Pose Estimation", CVPR 2022.
- Original repo: jasonqsy/Uni6D — uses RGB+RGB-D dual stream. RGB-only port
  drops the depth stream and uses the visual stream for direct 14 × 3 keypoint
  regression. Treated as Case R-.
- Implementation: `lineup/common/candidates/b3_uni6d.py`.

## Paper spec used
- Direct CNN regression of 6DoF pose (Uni6D doesn't use 2D-then-PnP — exactly
  the property required by Category B).
- Backbone: paper uses ResNet-50 + custom FPN; we use a **deeper backbone
  (ResNet-50)** as the architectural distinction from B1 REDE (ResNet-34).
- Head: global pool + MLP → 14 × 3 camera-frame coords.

## Architecture
- Backbone: torchvision ResNet-50 up to layer4 (B, 2048, H/32, W/32).
- AdaptiveAvgPool2d(8) → 2048 × 64 = 131072 features.
- MLP: 131072 → 1024 → 1024 → 42 (= 14 × 3).

## Losses
- Sym-aware MPJPE on 14 × 3 (`O_24`).
- Aux smooth-L1 on direct vertex correspondence.

## Modified files
- None outside `lineup/`.

## Deviations from paper
1. **RGB-only**: dropped paper's RGB-D dual stream.
2. **Direct regression** vs paper's intermediate "projection" head (paper
   claims "no projection breakdown"; our minimal RGB-only impl is the
   simplified visual branch).
3. AdamW vs Adam.

## Category B fitness
- [x] 14 × 3 camera-frame direct.
- [x] No 2D + PnP/lift.
- [x] Procrustes decode.
- [x] RGB-only.

Signed: AGENT 2026-04-25.
