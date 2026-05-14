# C3 FoundationPose — Code Audit (Case R-)

## Repo provenance
- Source paper: Wen et al., "FoundationPose: Unified 6D Pose Estimation and
  Tracking of Novel Objects", CVPR 2024 Highlight.
- Official repo: NVlabs/FoundationPose — pytorch3d, nvdiffrast, custom
  rendering ops; not buildable in our env. Treated as Case R-.
- Implementation: `lineup/common/candidates/c3_foundationpose.py`.

## Paper spec used (CVPR 2024)
- **RGB-D + CAD foundation**: paper uses a render-and-compare iterative
  refiner trained on synthetic RGB-D + CAD pairs.
- **Depth conditioning**: depth channel concatenated to RGB → 4-channel
  input. We synthesize depth via NOCS rasterizer's z-buffer (orthographic).
- **CAD branch**: canonical mesh embedded in MLP, fused with image features.
- **Refiner head**: paper iteratively refines pose; we use single-pass
  regression for overfit-1 capability test.

## Architecture
- Image branch: ImageNet pretrained ResNet-34 with **first conv inflated to
  4 input channels** (3 RGB + 1 depth). New depth-channel weight initialised
  to zero so the network behaves identically to RGB-only at init.
- Depth: synthetic per-instance depth normalised to [0, 1] (loaded from
  `phase0_C/depth/overfit/`).
- Mesh branch: 14 × 3 → MLP (42→128→128).
- Fusion + head: 1024 ⊕ 128 → MLP (1152→512→6+3+1).

## Losses
- L_R + L_t + L_s + L_PM. Same as C1, with `L_s` weight 1.0 (lesson from C1).

## Modified files
- None outside `lineup/`.

## Deviations from paper
1. **Synthetic depth, not real RGB-D**: derived from canonical mesh + GT pose,
   matching paper's training-time depth supervision form.
2. **Single-pass regression**, not iterative refiner.
3. **No paper foundation weights** — ImageNet-pretrained ResNet-34 substitute.

## Category C fitness
- [x] Pretrained large model (ImageNet ResNet-34).
- [x] CAD/canonical mesh input.
- [x] Depth conditioning (paper's signature feature).
- [x] Overfit via fine-tune.

Signed: AGENT 2026-04-25.
