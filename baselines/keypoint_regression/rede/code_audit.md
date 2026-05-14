# B1 REDE — Code Audit (Case R-)

## Repo provenance
- Source paper: Hua et al., "REDE: End-to-end Object 6D Pose Robust Estimation Using Differentiable Outliers Elimination", RA-L 2021.
- Official repo: https://github.com/HuaHuaY/REDE — paper requires RGB-D and uses
  custom ops; under torch 2.8 / CUDA 13.0 the build chain replicates the
  GDR-Net/SC6D-R+ blockers (mmcv-era + custom CUDA kernels). Treated as R-.
- Implementation file: `lineup/common/candidates/b1_rede.py`.

## Paper spec used (RA-L 2021)

### Input
- Original: RGB-D crop. **RGB-only port**: drop depth channel; rely on RGB
  feature for 3D keypoint regression. Documented deviation.

### Architecture (paper Sec III)
- **Backbone**: ResNet (paper uses ResNet variants); we use ResNet-34 to match
  A1/A2 backbones for fair compute budget.
- **3D keypoint head**: paper outputs per-vertex offset (radial vector field)
  followed by differentiable averaging/voting to get 14 × 3 camera-frame points.
- Our minimal RGB-only impl: ResNet-34 → AdaptiveAvgPool → MLP → 14 × 3 = 42
  outputs (camera-frame 3D coords directly). Voting is collapsed into the FC
  output for the overfit-1 capability test (paper's voting reduces variance,
  not the supervision form). Documented.

### Losses (paper Sec III.D)
- Symmetry-aware MPJPE: `min_{G ∈ sym} (1/14) Σ ‖p_pred_i − (s_gt R_gt G x_i + t_gt)‖`.
  We use the rhombic dodecahedron's `O_24` group from `common.geometry`.
- Auxiliary L2 on (R, t, s) decoded via Procrustes from `pred_kp_3d` ↔ canonical
  vertices (forces consistency between point regression and pose decode).

### Optimizer
- Paper: Adam. We use AdamW (consistent with A1/A2 deviations).

## Modified files
- None outside `lineup/`. Repo not imported.

## Deviations from paper (explicit)
1. **RGB-only**: depth channel dropped (paper assumes RGB-D). The 3D keypoint
   prediction now relies solely on visual cues; we add a tz-bias-free architecture
   (output is in camera frame directly).
2. **Differentiable voting → MLP regression**: paper's per-pixel offsets +
   averaging is collapsed into a global pooled MLP. The output type (14 × 3
   camera-frame coords) is preserved, satisfying the B-category constraint.
3. **Adam → AdamW** (same as A1/A2).

## Category B fitness (must all be YES)
- [x] Output: camera-frame 3D point coords (14 × 3) — direct regression.
- [x] No 2D keypoint + PnP/lift used anywhere.
- [x] Pose decode: Procrustes/Kabsch via `common.geometry.umeyama_np`.
- [x] RGB-only port feasible (this implementation IS the port).

Signed: AGENT 2026-04-25.
