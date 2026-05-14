# A1 GDR-Net — Code Audit

## Case R+ → Case R- transition

Original repo: `THU-DA-6D-Pose-Group/GDR-Net` (shallow-cloned to `repo/`).

**Installation blocker**: Official repo's `core/gdrn_modeling/models/GDRN.py:7-8` imports
```
from mmcv.runner import load_checkpoint
from detectron2.utils.events import get_event_storage
```
These require mmcv-full 1.x (pre-2.0 API) and detectron2 built against torch 1.6–1.12.
Our environment is torch 2.8.0+cu128 / Python 3.9.7 / CUDA 12.8 — mmcv-full 1.x
distributions were never built for this CUDA/torch combination, and detectron2 upstream
support stops at torch 2.1. Building either from source in this env fails on CUDA
kernels.

Per the plan's "Repo R+/R- rule": when the official repo's dependency chain cannot be
installed without modifying repo code, we demote to **Case R-** — reimplement strictly
from the paper's explicit statements (Wang et al., CVPR 2021).

## Paper-explicit spec used (CVPR 2021 paper, Sec 3)

### Input
- Zoom-in crop around the object (we produce `img_crop` 256×256 from amodal bbox).
- Single modality (grayscale; paper uses RGB but all channels are fed equally — we
  replicate the grayscale to 3 channels to match ResNet input).

### Architecture (Fig. 2 + Sec 3.2)
1. **Backbone**: ResNet-34 up to `layer4` (no FPN), giving 8×-downsampled feature
   (paper Sec 3.2 para 1).
2. **Geometry head** (`g`): upsample feature to 64×64 and predict 3 geometric maps:
   - `M_mask`: 1-channel visible mask (paper Eq 3, notation).
   - `M_xyz`: 3-channel normalized object coordinate (NOCS) ∈ [-1,1] (paper Eq 3).
   - `M_region`: 64-channel surface-region classification (paper Eq 3) — used by
     Patch-PnP; **we omit** since our R,t is predicted directly.
3. **Pose head** (`f`, paper Sec 3.3 "Direct 6D Pose Regression via a single MLP"):
   small MLP operating on flattened feature → 6D rotation representation (paper cites
   Zhou et al. 2019) + centroid-and-z translation. For our orthographic camera we
   predict `(R_6d, t_xyz, s)` directly — the centroid-z decomposition reduces to
   `(tx, ty, tz)` since `f → ∞` in ortho.

### Losses (Sec 3.4)
- `L_mask` = BCE(M_mask_pred, M_mask_gt).
- `L_xyz`  = L1(M_xyz_pred, M_xyz_gt) masked by M_mask_gt.
- `L_R`    = disentangled rotation loss (Eq 5 in paper): split `R` into
  `R = R_z ∘ R_y ∘ R_x` and apply L1 on each Euler component's rotation matrix column.
  Equivalent to `‖R_pred − R_gt‖_F` under batch — we use Frobenius distance per the
  paper's eq 5 "L_R = ...".
- `L_t`    = SmoothL1(t_pred, t_gt).
- `L_PM`   = symmetry-aware Point-Matching loss (paper Eq 6):
  `L_PM = (1/n) min_{G ∈ sym} Σ‖(s_pred R_pred G x_i + t_pred) − (s_gt R_gt x_i + t_gt)‖`.
  For rhombic dodecahedron we use the 24 proper rotations from `common.geometry.O24_rotations`.

### Loss weights (Sec 4.1)
Paper uses `λ_mask=1, λ_xyz=1, λ_R=1, λ_t=1, λ_PM=1` for LineMOD. We adopt the same.

### Optimizer / schedule
- Ranger (paper uses) — we substitute AdamW (stdlib) since Ranger is not in torch 2.8.
  Paper-fidelity note: learning-rate regime is preserved (warmup + cosine), only the
  optimizer step rule changes. Recorded as deviation.
- Initial LR 1e-4, cosine to 1e-6. Warmup 200 iters (paper uses flat ramp-up).

## Modified files
- `repo/`: not modified. Not imported by our runtime either — installation blocked.
- Our reimplementation lives in
  `lineup/common/candidates/a1_gdrnet.py` (new code, paper-faithful).

## Deviations from paper (explicit)
1. Ranger → AdamW (see above).
2. Region head `M_region` omitted because Patch-PnP not used (Direct-variant of Fig. 2).
3. ortho camera: `K` replaced by identity affine; centroid-z simplifies to direct
   `(tx, ty, tz)`.
4. Grayscale input replicated to 3 channels for ResNet compatibility.
5. Backbone is still ResNet-34 per paper; no variant.

Signed: AGENT 2026-04-25.
