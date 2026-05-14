# A2 SC6D — Code Audit (Case R-)

## Repo provenance
- Source paper: Cai et al., "SC6D: Symmetry-agnostic and Correspondence-free 6D Object Pose Estimation", 3DV 2022.
- Official repo: https://github.com/dingdingcai/SC6D-pose — **not used** (R+ blocked: depends on `pytorch3d` and `detectron2` with custom CUDA ops that fail to build under torch 2.8 / cu128 / system nvcc 13.0). Treated identically to A1's R- transition.
- Implementation file: `lineup/common/candidates/a2_sc6d.py` (new code, paper-based).

## Paper spec used (Cai et al., 3DV 2022)

### Input
- Object-centric crop, RGB → ResNet input (we replicate grayscale to 3 channels).

### Architecture (paper Sec 3 + Fig 2)
- **Backbone**: ResNet-34 (paper baseline; same as our A1 backbone).
- **Direct pose head**: a fully-connected MLP that operates on the global pooled
  feature and outputs (rotation, translation, scale) directly. SC6D's defining
  property is *no dense 2D-3D correspondence map* — only image-level pose
  regression with a symmetry-agnostic supervision. We use the 6D continuous
  rotation representation (Zhou et al. 2019) per paper's eq for SO(3) recovery.

### Losses (paper Sec 3.4)
- **Symmetry-agnostic Point Matching**: the same form as GDR-Net's `L_PM` —
  `L_PM = (1/n) min_{G ∈ sym} Σ ‖(s_pred R_pred G x_i + t_pred) − (s_gt R_gt x_i + t_gt)‖`.
  Paper relies on this "match the closest symmetry-equivalent pose" formulation
  to handle ambiguity without a precomputed symmetry-handling stream. We use
  the rhombic dodecahedron's 24-rotation group `O_24` from `common.geometry`.
- **Auxiliary R / t / scale L2 (or SmoothL1) losses** as supervision aids
  (paper Eq describes optional direct regression terms).

### Loss weights
- Paper uses uniform weights for these terms. Our overfit run uses
  `L = L_PM + L_R + 0.1 L_t + 0.1 L_scale`.

### Optimizer / schedule
- Paper trains with Adam, cosine LR. We use AdamW (consistent with A1 deviation),
  linear warm-up 200 iters → cosine to 1e-6 over the remainder.

## Modified files
- None outside `lineup/`. Repo not imported.

## Deviations from paper (explicit)
1. **Adam → AdamW** (same justification as A1; better weight-decay treatment in
   torch 2.8).
2. **No "scale-invariant translation" decomposition** — paper splits translation
   into image-plane center + log-z under a perspective camera. Our orthographic
   camera makes the decomposition degenerate, so we directly regress `(tx, ty, tz)`.
3. No region head (paper does not use one — but we explicitly note this since
   our A1 reimpl had a NOCS head; here we don't).

## Differentiation from A1 GDR-Net (Case R-) re-impl
- A1 has a NOCS map head + mask head + dense `L_xyz / L_mask` supervision in
  addition to the pose head. SC6D has **none of those** — pose head only.
  This is the structural distinction between the two papers. Our A2 SC6D
  candidate inherits A1's `_ResNet34Backbone` + `PoseHead`, drops the geometry
  head entirely, and uses `sym_pm_loss` + direct R/t/s regression losses.

Signed: AGENT 2026-04-25.
