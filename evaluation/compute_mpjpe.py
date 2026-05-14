"""B 카테고리 (B1/B2/B3) sym-aware MPJPE 재계산.

기존 dump JSON의 kp3d_raw (B 모델 raw 14-vertex 출력) + gt_verts 사용.
새 inference 없음. lineup/ read-only import만.

Pipeline (overfit_runner_ddp.py::compute_metrics 와 동일):
  R_gt, t_gt, s_gt = umeyama_np(canon, gt_verts)
  for G in O_24:
      sym_gt = s_gt * (canon @ G.T) @ R_gt.T + t_gt
      errs.append(mean over 14 verts of ||kp3d_raw - sym_gt||)
  mpjpe = min(errs)

Usage:
    python compute_mpjpe.py --model B1_rede
    python compute_mpjpe.py --model B2_uni6d
    python compute_mpjpe.py --model B3_ffb6d
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

# read-only import only
sys.path.insert(0, "<PROJECT_ROOT>/baselines")
from common import geometry as geo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_utils import get_canonical_vertices


REPO_ROOT = "<PROJECT_ROOT>"
DUMP_DIR = os.path.join(REPO_ROOT, "edge_length_experiment", "results", "per_model")
LOG_PATH = os.path.join(REPO_ROOT, "edge_length_experiment", "master_log.txt")


def log(msg: str) -> None:
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(line + "\n")


def compute_one_mpjpe(kp3d: np.ndarray, gt_verts: np.ndarray, canon: np.ndarray,
                      O24: np.ndarray) -> float:
    """sym-aware MPJPE for 1 instance."""
    R_gt, t_gt, s_gt = geo.umeyama_np(canon, gt_verts)
    errs = np.empty(len(O24))
    for k, G in enumerate(O24):
        sym_gt = s_gt * (canon @ G.T) @ R_gt.T + t_gt   # (14, 3)
        errs[k] = float(np.linalg.norm(kp3d - sym_gt, axis=1).mean())
    return float(errs.min())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=["B1_rede", "B2_uni6d", "B3_ffb6d"])
    args = ap.parse_args()

    name = args.model
    dump_path = os.path.join(DUMP_DIR, f"{name}_dump.json")
    if not os.path.exists(dump_path):
        raise FileNotFoundError(dump_path)

    log(f"START {name} MPJPE re-compute")
    t0 = time.time()
    d = json.load(open(dump_path))
    recs = d["records"]
    n = len(recs)

    canon = get_canonical_vertices().astype(np.float64)
    O24 = geo.O24_rotations()  # (24, 3, 3)

    mpjpes = np.empty(n)
    for i, r in enumerate(recs):
        kp3d = np.asarray(r["kp3d_raw"], dtype=np.float64)
        gt = np.asarray(r["gt_verts"], dtype=np.float64)
        mpjpes[i] = compute_one_mpjpe(kp3d, gt, canon, O24)
        if (i + 1) % 1000 == 0:
            print(f"  {i + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)

    payload = {
        "name": name,
        "source": "kp3d_raw (B model raw 14-vertex output, sym-aware over O_24)",
        "n": n,
        "mpjpe_mean": float(mpjpes.mean()),
        "mpjpe_median": float(np.median(mpjpes)),
        "mpjpe_rmse": float(np.sqrt((mpjpes ** 2).mean())),
        "mpjpe_min": float(mpjpes.min()),
        "mpjpe_max": float(mpjpes.max()),
        "per_instance_mpjpe": mpjpes.tolist(),
        "elapsed_s": time.time() - t0,
    }
    out_path = os.path.join(DUMP_DIR, f"{name}_mpjpe.json")
    with open(out_path, "w") as f:
        json.dump(payload, f)

    elapsed = time.time() - t0
    log(f"END {name} MPJPE n={n} mean={payload['mpjpe_mean']:.4f} "
        f"median={payload['mpjpe_median']:.4f} rmse={payload['mpjpe_rmse']:.4f} elapsed={elapsed:.0f}s")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
