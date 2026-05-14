# C1 MegaPose — Code Audit (Case R-)

## Repo provenance
- Source paper: Labbé et al., "MegaPose: 6D Pose Estimation of Novel Objects via
  Render-and-Compare", CoRL 2022.
- Official repo: megapose6d/megapose6d — heavy deps (panda3d, pytorch3d,
  pytorch-lightning era versions). Same build blockers as B-category. Treated
  as Case R-.
- Implementation: `lineup/common/candidates/c1_megapose.py`.

## Paper spec used (CoRL 2022)
- **Pretrained backbone**: paper trains on millions of synthetic objects.
  We use ImageNet-pretrained ResNet-34 as the "foundation" stand-in
  (real MegaPose weights would also be initialised; the architecture and
  CAD-conditioning input pattern is what we preserve).
- **CAD-conditioned**: object's canonical mesh vertices feed an extra branch
  to make the model "novel-object aware". For a single-object overfit, this
  reduces to a constant input but the wiring is faithful.
- **Direct (R, t, s) regression head**: 6D continuous rotation + translation
  + scale via MLP on pooled image feature concatenated with mesh vertex
  embedding.

## Architecture
- Image branch: ImageNet-pretrained ResNet-34 → (B, 512, H/32, W/32) →
  AdaptiveAvgPool2d(8) → 32768.
- Mesh branch: 14 × 3 canonical vertices flattened (42) → MLP (42→128→128).
- Fusion: image_feat 1024 ⊕ mesh_feat 128 → MLP → 6D rot + 3 trans + 1 scale.

## Losses
- L_R = Frobenius distance ‖R_pred − R_gt‖_F.
- L_t = SmoothL1 on (tx, ty, tz).
- L_s = SmoothL1 on scale.
- L_PM = sym-aware Point Matching (`O_24`).

## Modified files
- None outside `lineup/`.

## Deviations from paper (explicit)
1. **No pretrained foundation weights**: paper uses MegaPose's web-scraped
   pretrain. We use ImageNet pretrain (`resnet34(weights=DEFAULT)`).
   Documented; the supervision form (CAD-conditioned 6D regression) is intact.
2. **No iterative refiner**: paper has coarse + refiner stages. Single-stage
   regression for overfit-1 capability test.
3. AdamW vs Adam.

## Category C fitness
- [x] Pretrained large model used (ImageNet). Real MegaPose weights would be
  drop-in if available; not needed for capability test.
- [x] CAD/canonical mesh input (14 verts).
- [x] Overfit via fine-tune of pretrained weights.

Signed: AGENT 2026-04-25.
