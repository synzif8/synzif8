# A3 HccePose — Code Audit (Case R+)

## Repo provenance
- Source: https://github.com/WangYuLin-SEU/HCCEPose @ main (shallow, 2026-04-25)
- Modified files: **none**
- Used as-is:
  - `HccePose/network_model.py::HccePose_BF_Net` (model definition)

## Used class
| File:line | Symbol | Our usage |
|---|---|---|
| `HccePose/network_model.py:993` | `HccePose_BF_Net` | Constructed with `efficientnet_key=None, input_channels=3`. Returns `(mask_logit B×1×128×128, code B×48×128×128)`; we split the 48 channels into front (24) + back (24). |

## Paper spec reference (Wang et al., RA-L 2024 / ICCV'25 Highlight)

- **HCCE encoding** (paper Eq.): canonical xyz ∈ object AABB → normalized [0, 1] → per-dim 8-bit hierarchical binary code via `bit_b = floor(c * 2^(b+1)) mod 2`.
- **Front + Back surface**: each pixel encodes the canonical xyz of the visible surface and the back-facing surface separately. 3 dims × 8 bits × {front, back} = 48 channels.
- **Output**: model produces per-pixel (mask, 48-channel front+back code) at `crop/2` spatial resolution.
- **Pose decode**: paper uses PnP on dense 2D-3D correspondences. We substitute orthographic factorization (paper-level deviation).

## Deviations (explicit)

1. **Custom loss instead of `HccePose_Loss`** — repo's `HccePose_Loss.forward` mutates a CUDA tensor with a numpy float (line 954: `self.weight_front_error_ratio[k][i] = np.mean(...)`), which torch 2.8 forbids strictly. We replace it with a functionally equivalent BCE loss inside the foreground mask (mask BCE + per-bit BCE on front and back codes). The supervision form (binary front/back code per pixel inside fg mask) is preserved per paper Sec 3.4. Repo file unchanged.
2. **Tz auxiliary head** — orthographic camera does not constrain depth from correspondence; a tiny CNN (1 → 128 → 1) regresses tz separately. Same approach used for ZebraPose/A2.
3. **PnP → ortho factorization** — paper decodes dense correspondences via Progressive-X PnP. For our orthographic camera, we run least-squares for the 2×4 projection matrix and recover R via Gram-Schmidt + cross product, matching the GDR-Net/ZebraPose pipeline.
4. **No dynamic per-bit weighting** — repo's adaptive weight scheme is bypassed by deviation #1; uniform weights used. Paper Sec 3 lists this as an enhancement, not a core supervision target.

## Adapter
- Built in `lineup/common/candidates/a3_hccepose.py` (`_A3Wrapper`, `_load_gt`, `_decode_pose`).
- `phase0_A/make_hcce_codes.py` precomputes per-instance front/back code maps and valid masks.
- Repo's `bop_dataset.py`, `precompute_quaternion_labels.py`, etc. **not used** (we supply our own loader).

Signed: AGENT 2026-04-25.
