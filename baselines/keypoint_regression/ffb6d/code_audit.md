# B3 FFB6D RGB-only port — Code Audit (Case R-)

## Repo provenance
- Source paper: He et al., "FFB6D: A Full Flow Bidirectional Fusion Network for
  6D Pose Estimation", CVPR 2021.
- Original repo: ethnhe/FFB6D — RGB + depth bidirectional fusion via PointNet++.
  Same CUDA-build blockers as PVN3D (custom point-cloud ops, RandLA-Net).
  Treated as Case R- with RGB-only port.
- Implementation: `lineup/common/candidates/b3_ffb6d.py`.

## Paper spec used
- 3D keypoint direct regression (paper Sec 3.4): per-vertex 3D camera-frame
  coords from full-flow visual feature fusion.
- Backbone: paper's bidirectional fusion is collapsed (no depth) into a
  single-stream **ResNet-18** with global pool + MLP head — chosen ResNet-18
  to architecturally distinguish from B1 REDE (ResNet-34) and B2 Uni6D
  (ResNet-50).

## Architecture
- Backbone: torchvision ResNet-18 up to layer4 (B, 512, H/32, W/32).
- AdaptiveAvgPool2d(8) → 512 × 64 = 32768 features.
- MLP: 32768 → 768 → 768 → 42 (= 14 × 3).

## Losses
- Sym-aware MPJPE (`O_24`).
- Aux smooth-L1 on direct vertex correspondence.

## Modified files
- None outside `lineup/`.

## Deviations from paper
1. **RGB-only**: dropped depth + PointNet++ fusion.
2. **Single-stream**: paper's bidirectional fusion collapses to one path
   without depth. Output paradigm (14 × 3 camera-frame coords) preserved.
3. AdamW vs Adam.

## Category B fitness
- [x] 14 × 3 camera-frame direct.
- [x] No 2D + PnP/lift.
- [x] Procrustes decode.
- [x] RGB-only.

## Differentiation matrix
| Slot | Model | Backbone | Distinct property |
|---|---|---|---|
| B1 | REDE | ResNet-34 | global pool MLP, RGB-only port |
| B2 | Uni6D | ResNet-50 | deeper backbone, same head paradigm |
| B3 | FFB6D | ResNet-18 | shallower backbone, single-stream |

Signed: AGENT 2026-04-25.
