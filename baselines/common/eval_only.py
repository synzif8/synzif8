"""Re-evaluate a trained candidate from saved weights without re-training.

Rebuilds model, loads weights_last.pt, runs forward_eval on overfit split,
re-writes metrics.json + viz/ + STATUS.md.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import cv2
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.candidates import load_candidate
from common.dataset_instance import InstanceDataset
from common import geometry as geo
from common import viz_no_leak as vnl
from common.overfit_runner_ddp import (
    _decode_preds, compute_metrics, aggregate_metrics, tensor_to_bgr,
    render_pred_grid, PASS_THRESH,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--split", default="<PROJECT_ROOT>/baselines/phase0_common/splits/overfit.txt")
    ap.add_argument("--weights", default=None, help="override; else <out-dir>/weights_last.pt")
    args = ap.parse_args()

    device = torch.device("cuda:0")
    candidate = load_candidate(args.model)
    model = candidate.build_model(device)
    w_path = args.weights or os.path.join(args.out_dir, "weights_last.pt")
    state = torch.load(w_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()

    crop_size = candidate.crop_size
    ds = InstanceDataset(split_path=args.split, crop_size=crop_size)
    viz_dir = os.path.join(args.out_dir, "viz")
    os.makedirs(viz_dir, exist_ok=True)
    # clear stale viz so we don't mix
    for fn in os.listdir(viz_dir):
        if fn.endswith(".png"):
            os.remove(os.path.join(viz_dir, fn))

    all_recs = []
    canonical_np = None
    with torch.no_grad():
        for i in range(len(ds)):
            sample = ds[i]
            sample_d = {k: (v.unsqueeze(0).to(device) if torch.is_tensor(v) else v) for k, v in sample.items()}
            eo = candidate.forward_eval(model, sample_d)
            preds = _decode_preds(eo, sample_d)
            rec = compute_metrics(preds, sample_d)[0]
            all_recs.append(rec)
            if canonical_np is None:
                canonical_np = sample["vertices_obj"].cpu().numpy()
            crop_affine_np = sample["crop_affine"].cpu().numpy()
            preds_i = {"R": preds["R"][0], "t": preds["t"][0], "s": float(preds["s"][0])}
            grid = render_pred_grid(sample, preds_i, canonical_np, crop_affine_np)
            cv2.imwrite(os.path.join(viz_dir, f"overfit_{int(sample['scene_id'].item()):04d}_{int(sample['obj_id'].item()):04d}.png"), grid)

    agg = aggregate_metrics(all_recs)
    thresh = PASS_THRESH[candidate.category]
    numeric_pass = all(agg.get(k, 1e9) <= thresh[k] for k in thresh)

    # update metrics.json — keep training logs if file exists
    metrics_path = os.path.join(args.out_dir, "metrics.json")
    if os.path.exists(metrics_path):
        m = json.load(open(metrics_path))
    else:
        m = {}
    m["aggregate"] = agg
    m["per_instance"] = all_recs
    m["threshold"] = thresh
    m["numeric_pass"] = bool(numeric_pass)
    m["eval_only_reeval"] = True
    with open(metrics_path, "w") as f:
        json.dump(m, f, indent=2)

    with open(os.path.join(args.out_dir, "STATUS.md"), "w") as f:
        f.write(f"# {candidate.name} — overfit status (decoder-fix eval-only rerun)\n\n")
        f.write(f"- category: {candidate.category}\n")
        f.write(f"- numeric_gate: {'PASS' if numeric_pass else 'FAIL'}\n")
        f.write(f"- VISUAL_PASS: PENDING (reviewer must inspect viz/)\n\n")
        f.write("## Aggregate metrics\n")
        for k, v in agg.items():
            f.write(f"- {k}: {v:.4f}\n")
        f.write(f"\nthreshold: {thresh}\n")
    print("re-eval aggregate:", agg)
    print("numeric_pass:", numeric_pass)


if __name__ == "__main__":
    main()
