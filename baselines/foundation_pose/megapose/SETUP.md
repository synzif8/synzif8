# C1 — MegaPose

This directory contains only the SynZIF-8 candidate wrapper (`c1_megapose_official.py`).
The upstream MegaPose-6D codebase is required for inference and must be cloned
separately.

## Setup

```bash
# from <PROJECT_ROOT>/baselines/foundation_pose/megapose/
git clone https://github.com/megapose6d/megapose6d.git external
cd external

# Install per upstream README:
#   - conda env from environment.yml
#   - python -m pip install -e .

# Download pretrained checkpoints into external/local_data/ per upstream README.
```

The candidate wrapper (`c1_megapose_official.py`) expects the upstream MegaPose
code to be importable from `external/src/megapose/...` and the pretrained
weights at the standard upstream `local_data/` location.

Original paper:
- Labbé et al., **MegaPose: 6D Pose Estimation of Novel Objects via Render-and-Compare**, CoRL 2022.
- License: see upstream repo (BSD-3-Clause as of writing).
