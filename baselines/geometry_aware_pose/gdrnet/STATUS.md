# A1 — GDR-Net (Wang et al., CVPR 2021)

Status: **IN_PROGRESS** (M1 fitness)

## M1. Category-fitness checklist — Category A (appearance-based 6D pose)

- [x] 출력이 (R, t) 또는 dense correspondence → PnP/Procrustes로 (R, t) 복원 가능
      → GDR-Net outputs dense 2D-3D correspondence (NOCS map) + direct R,t regression via Patch-PnP.
- [x] 학습 loss가 pose 관련 (rotation/translation/correspondence)
      → L_pose = L_R (disentangled allocentric) + L_trans + L_NOCS + L_mask.
- [x] RGB(또는 grayscale) 단일 모달 입력 허용
      → 입력은 RGB crop 256×256. Grayscale은 3채널 복제로 적용.
- [x] 3D keypoint 직접 회귀 방식이 아님 → Category A 적합.

## Publication check
- Venue: CVPR 2021. **Published ✅**. Not preprint-only.

## Planned: M2. Clone
- Repo candidate: https://github.com/THU-DA-6D-Pose-Group/GDR-Net

Signed: AGENT 2026-04-25.
