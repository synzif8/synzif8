# SynZIF-8 — Code Release

This repository accompanies the NeurIPS 2026 submission **"SynZIF-8"** and
contains all code needed to reproduce the dataset, baselines, and evaluations
described in the paper.

**📦 Dataset on Hugging Face:**
[**huggingface.co/datasets/synzif8/SynZIF-8**](https://huggingface.co/datasets/synzif8/SynZIF-8)

> **Note on placeholders.** Source paths in this repository use two placeholders
> instead of absolute paths:
> - `<PROJECT_ROOT>` — the path to this cloned repository.
> - `<DATA_ROOT>`    — the path where the rendered dataset (`dataset_v6/`,
>   `dataset_v6_sem/`, etc.) is stored. The dataset itself is released on
>   Hugging Face (link above); download and extract it to `<DATA_ROOT>` before
>   running any pipeline.
>
> After cloning, run `bash setup.sh` once and the placeholders are rewritten to
> your local absolute paths automatically. See [Quick start](#quick-start) below.

---

## Directory layout

```
.
├── data_generation/
│   ├── rendering/                3D rhombic-dodecahedron scene rendering,
│   │                             Gaussian clustering, orthographic projection,
│   │                             per-instance annotation extraction
│   │   ├── create_dataset_v6.py
│   │   └── splits/               train / val / test scene split manifests
│   │
│   └── stylization/              SEM stylization (SD 1.5 + LoRA + ControlNet)
│       ├── SEM_dev_70/           **placeholder** — 70 real SEM images you
│       │                         must provide yourself (see Sec 3.1 of paper
│       │                         and data_generation/README.md)
│       ├── exp21_sd15_lora/      LoRA fine-tuning
│       ├── exp22_sd15_controlnet/ ControlNet (Canny) fine-tuning
│       └── exp23_sd15_inference/ inference script
│
├── segmentation/                 Mask R-CNN front-end detector (seg_exp07)
│   ├── config.yaml
│   └── src/{train,evaluate,seg_dataset}.py
│
├── baselines/
│   ├── common/                   Unified DDP runner + geometry + dataset adapter
│   │   ├── overfit_runner_ddp.py
│   │   ├── full_runner_ddp.py
│   │   ├── geometry.py           sym-aware ADD-S / MPJPE, O_24, Umeyama
│   │   └── candidates/           Model definitions (registered by name)
│   │
│   ├── keypoint_regression/      Category B
│   │   ├── ffb6d/  rede/  uni6d/
│   │
│   ├── geometry_aware_pose/      Category A
│   │   ├── gdrnet/  sc6d/  hccepose/
│   │
│   └── foundation_pose/          Category C (external repos)
│       ├── megapose/  gigapose/  foundationpose/
│
├── evaluation/                   Edge-length RRMSE, MPJPE, Hungarian matching,
│                                 per-instance Z-offset correction
│   ├── edge_utils.py
│   ├── compute_mpjpe.py
│   ├── compute_metrics_with_std.py
│   ├── compute_pose_metrics_maskrcnn.py
│   ├── build_edge_d_p_ranking_two_sets.py
│   ├── eval_vertex_annotator*.py
│   ├── lineup_scripts/real_eval/    real-SEM evaluation pipeline
│   └── scripts_5_4z_depth/          Z-offset analysis (Sec. 5.4 of the paper)
│
├── annotation_tool/              FastAPI web app for manual edge / vertex
│                                 annotation on SEM images
│   ├── server.py                 vertex annotation server
│   ├── server_length.py          edge-length annotation server
│   └── static/                   browser front-end
│
└── human_perception_study/       Notebooks for the N=6 domain-expert
                                  Real-vs-Synthetic perception study
    ├── human_exp_Real_Fake.ipynb
    ├── human_exp_Edge_esti.ipynb
    └── human_perception_analysis.ipynb
```

---

## Quick start

1. Clone this repository.
2. Download the dataset (rendered + stylized) from
   [huggingface.co/datasets/synzif8/SynZIF-8](https://huggingface.co/datasets/synzif8/SynZIF-8)
   and extract it to a directory of your choice (call it `DATA_DIR`).
3. Rewrite path placeholders to your machine's absolute paths:
   ```bash
   bash setup.sh /absolute/path/to/DATA_DIR
   ```
   This replaces `<PROJECT_ROOT>` with the repo path and `<DATA_ROOT>` with
   `DATA_DIR` across all source files.
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
   (Foundation-pose baselines have additional per-repo requirements in their
   `external/` folders.)
5. To run a baseline overfit smoke-test:
   ```bash
   python baselines/common/overfit_runner_ddp.py --model A1_gdrnet \
          --out-dir runs/A1_gdrnet
   ```

---

## License

This repository combines original code (Apache-2.0 unless stated otherwise) with
several third-party baseline repositories that retain their original licenses
in their respective subdirectories. See each `baselines/foundation_pose/*/external/LICENSE`
and `baselines/geometry_aware_pose/hccepose/repo/LICENSE`.
