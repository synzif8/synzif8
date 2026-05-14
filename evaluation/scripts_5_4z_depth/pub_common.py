"""Shared helpers for the publication-quality 5.4Z_depth figures.

Loads the single selected instance (scene_18225 obj_9, idx=1068) from each of
the 9 model dumps and exposes a stable color map / edge / face definition.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path("<PROJECT_ROOT>/evaluation")
PER_MODEL = ROOT / "results" / "per_model"
PUB = ROOT / "results" / "5.4Z_depth" / "visualizations" / "z_3d_publication"
DATASET = Path("<DATA_ROOT>/dataset_v6")
sys.path.insert(0, str(ROOT / "scripts"))
from edge_utils import compute_edge_lengths_3d, get_edge_pairs, get_canonical_vertices  # noqa: E402

EDGES = get_edge_pairs()
CANONICAL = get_canonical_vertices()
FACES = [
    [8, 5, 1, 6], [8, 6, 2, 9], [8, 9, 4, 7], [8, 7, 3, 5],
    [0, 10, 1, 11], [0, 11, 2, 13], [0, 13, 4, 12], [0, 12, 3, 10],
    [1, 5, 3, 10], [3, 7, 4, 12], [4, 9, 2, 13], [2, 6, 1, 11],
]
TRI_FACES = []
for q in FACES:
    TRI_FACES.append([q[0], q[1], q[2]])
    TRI_FACES.append([q[0], q[2], q[3]])
TRI_FACES = np.array(TRI_FACES, dtype=np.int64)

VERTEX_KIND = {0: "bottom apex", 8: "top apex"}
for i in (1, 2, 3, 4):
    VERTEX_KIND[i] = "axial"
for i in (5, 6, 7, 9, 10, 11, 12, 13):
    VERTEX_KIND[i] = "corner"

VERTEX_KIND_COLOR = {
    "bottom apex": "#1f77b4",
    "top apex": "#d62728",
    "axial": "#2ca02c",
    "corner": "#ff7f0e",
}

ENTRIES = [
    ("01_A1_gdrnet", "A1_gdrnet", "A1 GDR-Net", "A", "pred_verts", "#1f77b4"),
    ("02_A2_sc6d", "A2_sc6d", "A2 SC6D", "A", "pred_verts", "#4a90c2"),
    ("03_A3_hccepose", "A3_hccepose", "A3 HccePose", "A", "pred_verts", "#6ba8d0"),
    ("04_B1_rede_raw", "B1_rede", "B1 REDE (raw)", "B", "kp3d_raw", "#ff7f0e"),
    ("05_B2_uni6d_raw", "B2_uni6d", "B2 Uni6D (raw)", "B", "kp3d_raw", "#ffa040"),
    ("06_B3_ffb6d_raw", "B3_ffb6d", "B3 FFB6D (raw)", "B", "kp3d_raw", "#ffc575"),
    ("07_C1_megapose_official", "C1_megapose_official", "C1 MegaPose", "C", "pred_verts", "#2ca02c"),
    ("08_C2_gigapose_official", "C2_gigapose_official", "C2 GigaPose", "C", "pred_verts", "#5cb85c"),
    ("09_C3_foundationpose_official", "C3_foundationpose_official", "C3 FoundPose", "C", "pred_verts", "#8fdc8f"),
]

INFO_PATH = ROOT / "results" / "5.4Z_depth" / "_info" / "selected_instance.txt"
INFO = INFO_PATH.read_text()
IDX = int([l for l in INFO.splitlines() if "global index" in l][0].split(":")[-1])
SCENE_ID = int([l for l in INFO.splitlines() if "selected instance" in l][0].split("scene_")[-1].split()[0])
OBJ_ID = int([l for l in INFO.splitlines() if "selected instance" in l][0].split("obj_")[-1])

# Pose summary (carry over R_err / ADD-S etc. for each entry)
SUMMARY = json.load(open(ROOT / "results" / "summary_v3.json"))
_pose: dict[str, dict] = {}
for r in SUMMARY["rows"]:
    n = r["name"]
    if n.endswith("_zeroshot"):
        continue
    if n in _pose:
        if r["mape_mean"] > _pose[n]["mape_mean"]:
            _pose[n] = r
    else:
        _pose[n] = r


def per_inst_edge_metrics(gt: np.ndarray, pred: np.ndarray) -> dict:
    e_gt = compute_edge_lengths_3d(gt[None], EDGES)[0]
    e_pred = compute_edge_lengths_3d(pred[None], EDGES)[0]
    safe = np.where(np.abs(e_gt) > 1e-9, e_gt, 1.0)
    abs_err = np.abs(e_pred - e_gt)
    return {
        "edge_mae": float(abs_err.mean()),
        "edge_mape": float(np.nanmean(abs_err / np.abs(safe) * 100)),
    }


def load_instance() -> dict:
    rows = []
    gt = None
    for stem, dump_name, label, cat, field, color in ENTRIES:
        rec = json.load(open(PER_MODEL / f"{dump_name}_dump.json"))["records"][IDX]
        gt_i = np.array(rec["gt_verts"], dtype=np.float64)
        if gt is None:
            gt = gt_i
        else:
            assert np.allclose(gt, gt_i, atol=1e-6), f"GT mismatch for {dump_name}"
        pred = np.array(rec[field], dtype=np.float64)
        edge_m = per_inst_edge_metrics(gt_i, pred)
        diff = pred - gt_i
        rows.append({
            "stem": stem,
            "dump": dump_name,
            "label": label,
            "category": cat,
            "color": color,
            "pred": pred,
            "edge_mae": edge_m["edge_mae"],
            "edge_mape": edge_m["edge_mape"],
            "z_err_mean": float(np.abs(diff[:, 2]).mean()),
            "z_err_per_vert": np.abs(diff[:, 2]).tolist(),
            "x_err_mean": float(np.abs(diff[:, 0]).mean()),
            "y_err_mean": float(np.abs(diff[:, 1]).mean()),
            "gt_z_mean": float(gt_i[:, 2].mean()),
            "pred_z_mean": float(pred[:, 2].mean()),
            "add_s": _pose[dump_name]["add_s"],
            "r_err": _pose[dump_name]["r_err"],
            "t_err": _pose[dump_name]["t_err"],
            "t_z_err_proxy": float(np.abs(pred[:, 2].mean() - gt_i[:, 2].mean())),
        })
    return {"gt": gt, "rows": rows, "scene_id": SCENE_ID, "obj_id": OBJ_ID, "idx": IDX}


def save_instance_data(data: dict, path: Path) -> None:
    out = {
        "scene_id": data["scene_id"],
        "obj_id": data["obj_id"],
        "idx": data["idx"],
        "gt_verts": data["gt"].tolist(),
        "predictions": {
            r["dump"]: {
                "label": r["label"],
                "category": r["category"],
                "field": [e[4] for e in ENTRIES if e[1] == r["dump"]][0],
                "pred_verts": r["pred"].tolist(),
                "edge_mae": r["edge_mae"],
                "edge_mape": r["edge_mape"],
                "x_err_mean": r["x_err_mean"],
                "y_err_mean": r["y_err_mean"],
                "z_err_mean": r["z_err_mean"],
                "gt_z_mean": r["gt_z_mean"],
                "pred_z_mean": r["pred_z_mean"],
                "add_s": r["add_s"],
                "r_err": r["r_err"],
                "t_err": r["t_err"],
            }
            for r in data["rows"]
        },
        "metadata": {
            "diameter_canonical_px": 6.928203230275509,
            "n_vertices": 14, "n_edges": 24, "n_faces": 12,
            "gt_z_min": float(data["gt"][:, 2].min()),
            "gt_z_max": float(data["gt"][:, 2].max()),
            "gt_z_mean": float(data["gt"][:, 2].mean()),
            "gt_center": data["gt"].mean(axis=0).tolist(),
            "gt_bbox_size": (data["gt"].max(0) - data["gt"].min(0)).tolist(),
        },
    }
    path.write_text(json.dumps(out, indent=2))
