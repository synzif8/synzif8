# C3 — FoundationPose

This directory contains only the SynZIF-8 candidate wrapper (`c3_foundationpose_official.py`).
The upstream FoundationPose codebase is required for inference and must be cloned
separately.

## Setup

```bash
# from <PROJECT_ROOT>/baselines/foundation_pose/foundationpose/
git clone https://github.com/NVlabs/FoundationPose.git external
cd external

# Install per upstream README:
#   - build_all.sh (or build_all_conda.sh) to compile C++/CUDA extensions
#   - download weights into external/weights/ per upstream README

# Note: FoundationPose requires CUDA-compiled extensions (mycpp/, bundlesdf/mycuda/)
#       — build per upstream instructions before running inference.
```

The candidate wrapper (`c3_foundationpose_official.py`) expects the upstream
FoundationPose code to be importable from `external/` and the compiled
extensions to be available.

Original paper:
- Wen et al., **FoundationPose: Unified 6D Pose Estimation and Tracking of Novel
  Objects**, CVPR 2024.
- License: see upstream repo (NVIDIA Source Code License — Non-Commercial).
