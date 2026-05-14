"""9-model inference for the new length-annotation set.

Mirrors `scripts/real_eval/inference.py` but reads from the new
`5_5_realSEM_length_exp/realSEM_length20_annotation/` directory (each rhombic
has exactly 2 GT vertices = the endpoints of the annotated 2D edge).

Saves per-image NPZ to `runs/full/_length_eval/preds/<stem>.npz`.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from typing import Iterator

import cv2
import numpy as np
import torch

REPO_ROOT = "<PROJECT_ROOT>"
sys.path.insert(0, os.path.join(REPO_ROOT, "lineup"))

from common.candidates import load_candidate
from common.overfit_runner_ddp import _decode_preds
from scripts.real_eval.data import build_one, make_batch
from scripts.real_eval.inference import EVAL_QUEUE, project_verts_full, crop_project

GT_DIR = "<PROJECT_ROOT>/annotation_tool/5_5_realSEM_length_exp/realSEM_length20_annotation"
REAL_SEM_DIR = "<PROJECT_ROOT>/annotation_tool/sem_images"
OUT_ROOT = os.path.join(REPO_ROOT, "lineup/runs/full/_length_eval/preds")
RUNS_ROOT = os.path.join(REPO_ROOT, "lineup/runs/full")


def iter_image_annotations(limit: int | None = None) -> Iterator[tuple[str, dict, np.ndarray]]:
    files = sorted(f for f in os.listdir(GT_DIR) if f.endswith(".json"))
    for i, jf in enumerate(files):
        if limit is not None and i >= limit:
            break
        with open(os.path.join(GT_DIR, jf)) as f:
            ann = json.load(f)
        img_path = os.path.join(REAL_SEM_DIR, ann["image_name"])
        if not os.path.exists(img_path):
            print(f"[skip] missing image: {img_path}")
            continue
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[skip] cv2 failed: {img_path}")
            continue
        yield ann["image_name"], ann, img


def run(device: torch.device, limit_images: int | None, batch_size: int):
    os.makedirs(OUT_ROOT, exist_ok=True)
    print(f"[init] loading {len(EVAL_QUEUE)} candidates on {device}", flush=True)
    candidates = []
    for name, wrel in EVAL_QUEUE:
        cand = load_candidate(name)
        model = cand.build_model(device)
        wp = os.path.join(RUNS_ROOT, wrel)
        state = torch.load(wp, map_location=device, weights_only=True)
        try:
            model.load_state_dict(state, strict=True)
        except Exception as e:
            print(f"[{name}] strict-load failed ({e}); fallback to non-strict")
            model.load_state_dict(state, strict=False)
        model.eval()
        candidates.append((name, cand, model))
        print(f"  loaded {name}", flush=True)

    canonical = np.array(
        [[0,0,-2],[0,2,0],[-2,0,0],[2,0,0],[0,-2,0],[1,1,1],[-1,1,1],
         [1,-1,1],[0,0,2],[-1,-1,1],[1,1,-1],[-1,1,-1],[1,-1,-1],[-1,-1,-1]],
        dtype=np.float32,
    )

    for img_name, ann, img_full in iter_image_annotations(limit=limit_images):
        t0 = time.time()
        rhombics = ann["rhombics"]
        if not rhombics:
            print(f"[{img_name}] no rhombics; skip")
            continue
        samples = [build_one(img_name, r, img_full) for r in rhombics]
        N = len(samples)
        M = len(candidates)

        all_R = np.zeros((N, M, 3, 3), np.float32)
        all_t = np.zeros((N, M, 3),     np.float32)
        all_s = np.zeros((N, M),        np.float32)
        all_v_full = np.zeros((N, M, 14, 2), np.float32)
        all_v_crop = np.zeros((N, M, 14, 2), np.float32)
        all_kp3d = np.full((N, M, 14, 3), np.nan, np.float32)   # NaN if model is not B

        for mi, (name, cand, model) in enumerate(candidates):
            for s_start in range(0, N, batch_size):
                s_end = min(s_start + batch_size, N)
                chunk = samples[s_start:s_end]
                batch = make_batch(chunk, device)
                with torch.no_grad():
                    out = cand.forward_eval(model, batch)
                preds = _decode_preds(out, batch)
                R = preds["R"]; t = preds["t"]; s = preds["s"]
                v_full = project_verts_full(R, t, s, canonical)
                aff = batch["crop_affine"].cpu().numpy()
                v_crop = crop_project(v_full, aff)
                all_R[s_start:s_end, mi]      = R
                all_t[s_start:s_end, mi]      = t
                all_s[s_start:s_end, mi]      = s
                all_v_full[s_start:s_end, mi] = v_full
                all_v_crop[s_start:s_end, mi] = v_crop
                if preds.get("kp3d") is not None:
                    all_kp3d[s_start:s_end, mi] = preds["kp3d"]

        stem = os.path.splitext(img_name)[0]
        out_path = os.path.join(OUT_ROOT, f"{stem}.npz")
        # Also store p1, p2, length_px from each rhombic for the downstream length comparison.
        p1 = np.array([r["p1"] for r in rhombics], np.float32)              # (N, 2)
        p2 = np.array([r["p2"] for r in rhombics], np.float32)              # (N, 2)
        length_px = np.array([r["length_px"] for r in rhombics], np.float32)  # (N,)

        np.savez_compressed(
            out_path,
            image_name=np.array(img_name),
            seg_idx=np.array([s.seg_idx for s in samples], np.int32),
            bbox_full=np.array([s.bbox_full for s in samples], np.int32),
            crop_affine=np.stack([s.crop_affine for s in samples]),
            gt_full=np.array([s.gt_vertices_full for s in samples], dtype=object),
            gt_crop=np.array([s.gt_vertices_crop for s in samples], dtype=object),
            p1=p1, p2=p2, length_px=length_px,
            R=all_R, t=all_t, s=all_s,
            verts_2d_full=all_v_full,
            verts_2d_crop=all_v_crop,
            kp3d_raw=all_kp3d,
            model_names=np.array([n for n, _ in EVAL_QUEUE]),
        )
        elapsed = time.time() - t0
        print(f"[{img_name}] N={N} -> {out_path}   ({elapsed:.1f}s)", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-images", type=int, default=None)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="cuda:0")
    args = ap.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    run(device, args.limit_images, args.batch)


if __name__ == "__main__":
    main()
