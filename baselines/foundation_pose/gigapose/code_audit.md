# C2 GigaPose — Code Audit (Case R-)

## Repo provenance
- Source paper: Nguyen et al., "GigaPose: Fast and Robust Novel Object Pose
  Estimation via One Correspondence", CVPR 2024.
- Official repo: nv-nguyenho/gigaPose — pytorch3d, custom render-compare ops.
  Treated as Case R-.
- Implementation: `lineup/common/candidates/c2_gigapose.py`.

## Paper spec used (CVPR 2024)
- **DINOv2 / Vision Transformer foundation**: paper relies on a self-supervised
  ViT backbone for novel-object generalisation. We use ImageNet-pretrained
  ViT-B/16 from torchvision as the foundation visual backbone.
- **CAD-conditioned template features + 1-correspondence pose**: paper
  matches a single best canonical-template feature to image features. For
  our single-known-object overfit setting, the canonical mesh acts as the
  template; we compute a fixed template feature embedding (mesh MLP) and
  fuse with image features for pose head — preserving the "template feature
  + image feature → pose" structure.

## Architecture
- Image branch: ViT-B/16 ImageNet pretrained → CLS token (768).
- Mesh branch: 14 × 3 → MLP (42→128→128) → mesh feature.
- Fusion + head: cat(768, 128) → MLP (896→512→6+3+1).
- Output: rot 6D + trans + scale.

## Losses
- Same as C1: L_R + L_t + L_s + sym PM (`O_24`).

## Modified files
- None outside `lineup/`.

## Deviations from paper
1. **No DINOv2**: replaced with ImageNet pretrained ViT-B/16 (foundation
   visual transformer with public weights).
2. **No template renderer**: paper renders templates online; we use mesh MLP
   embedding as the static template feature for our single-class overfit.
3. AdamW vs Adam.

## Category C fitness
- [x] Pretrained foundation (ViT-B/16 ImageNet).
- [x] CAD/canonical mesh input.
- [x] Overfit via fine-tune of pretrained weights.

Signed: AGENT 2026-04-25.
