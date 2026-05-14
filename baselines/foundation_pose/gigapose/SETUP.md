# C2 — GigaPose

This directory contains only the SynZIF-8 candidate wrapper (`c2_gigapose_official.py`).
The upstream GigaPose codebase is required for inference and must be cloned
separately.

## Setup

```bash
# from <PROJECT_ROOT>/baselines/foundation_pose/gigapose/
git clone https://github.com/nv-nguyen/gigapose.git external
cd external

# Install per upstream README:
#   - conda env from environment.yml
#   - python -m pip install -e .

# Download pretrained checkpoints into external/pretrained/ per upstream README.
```

The candidate wrapper (`c2_gigapose_official.py`) expects the upstream GigaPose
code to be importable from `external/src/...` and the pretrained weights at
the standard upstream `pretrained/` location.

Original paper:
- Nguyen et al., **GigaPose: Fast and Robust Novel Object Pose Estimation via
  One Correspondence**, CVPR 2024.
- License: see upstream repo.
