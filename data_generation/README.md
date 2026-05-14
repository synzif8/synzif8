# Data Generation Pipeline

This directory contains both stages of the SynZIF-8 data generation pipeline
described in Section 3 of the paper:

```
data_generation/
├── rendering/           §3.2.1 — 3D rhombic-dodecahedron scene rendering
└── stylization/         §3.2.2 — SEM stylization (SD 1.5 + LoRA + ControlNet)
```

---

## Note on Real SEM Reference Data (`SEM_dev_70/`)

The LoRA + ControlNet fine-tuning in the stylization stage requires a set of
real SEM micrographs of ZIF-8 crystals (70 images, see **§3.1** and
**Appendix B** of the paper) as reference data.

**These reference images are *not* redistributed in this repository.** The
reasons are:

1. They are **co-owned with the chemistry collaborators** listed in the
   author roster (the synthesis and SEM imaging were performed by them).
2. The dataset's **Croissant metadata** on Hugging Face explicitly states
   that the real SEM reference images are *"not redistributed as part of
   this dataset"*.

Consequently, the directory `data_generation/stylization/SEM_dev_70/` is
intentionally absent from this release and is listed in `.gitignore`.

### How to run the stylization pipeline locally

You have two options:

#### Option A — Use the SynZIF-8 dataset directly (recommended)

If your goal is to **use** SynZIF-8 (e.g., to train a perception model or
to benchmark a baseline), you do **not** need to re-run the stylization
stage. The already-stylized SEM-style images are available on Hugging Face:

> https://huggingface.co/datasets/synzif8/SynZIF-8

#### Option B — Reproduce the stylization stage on your own SEM data

If your goal is to *reproduce* the stylization stage (e.g., to retrain
LoRA + ControlNet on a different chemistry / instrument):

1. Collect your own real SEM micrographs of ZIF-8 (or another crystal of
   interest). The synthesis recipe and SEM imaging parameters are fully
   documented in **Appendix B** of the paper, so independent collection of
   an equivalent reference set is supported.
2. Place 70 (or more) reference images at
   `data_generation/stylization/SEM_dev_70/*.png` at 1024 × 686 resolution.
3. Run the LoRA fine-tuning:
   ```bash
   cd data_generation/stylization/exp21_sd15_lora
   python train_and_infer.py
   ```
4. Run the ControlNet fine-tuning:
   ```bash
   cd ../exp22_sd15_controlnet
   python train_and_infer.py
   ```
5. Run inference (mask-guided contrast calibration is applied automatically):
   ```bash
   cd ../exp23_sd15_inference
   python inference_batch_v6.py
   ```

The expected input layout for the configs is `../SEM_dev_70` (relative path
from each `expXX_*` directory), as set in `exp21_sd15_lora/config.yaml`.
