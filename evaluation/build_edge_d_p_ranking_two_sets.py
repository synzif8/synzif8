"""Build edge / D / P ranking comparison tables for two evaluation sets.

(1) 500-scene test split: Mask R-CNN-based dumps (n=11,375 pairs), 12 BASELINES (no human).
(2) 407-instance human-aligned: vertex_20img dumps filtered by Hungarian human matches,
    13 entries (12 BASELINES + vertex_annotator human).

Output: results/edge_d_p_ranking_two_sets.md
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_utils import compute_edge_lengths_3d, get_edge_pairs

ROOT = "<PROJECT_ROOT>/evaluation"
RESULTS_DIR = os.path.join(ROOT, "results")
PER_MODEL_DIR = os.path.join(RESULTS_DIR, "per_model")
DATASET_DIR = "<DATA_ROOT>/dataset_v6"
VA_PRED_DIR = "<PROJECT_ROOT>/annotation_tool/length_annotations"
VA_GT_DIR = "<PROJECT_ROOT>/annotation_tool/sem_images_json"
ALIGNED_JSON = os.path.join(RESULTS_DIR, "vertex_annotator_eval_20img_aligned.json")

# Visibility threshold for Set A-2 (GT-crop + fully visible filter)
SET_A2_MIN_VISIBILITY = 1.0  # strict: fully visible only
SET_A2_LABEL = "fully visible (visibility = 1.0)"

MODELS = [
    ("A1_gdrnet", "A1 GDR-Net", "A"),
    ("A2_sc6d", "A2 SC6D", "A"),
    ("A3_hccepose", "A3 HccePose", "A"),
    ("B1_rede", "B1 REDE (raw 24)", "B"),
    ("B2_uni6d", "B2 Uni6D (raw 24)", "B"),
    ("B3_ffb6d", "B3 FFB6D (raw 24)", "B"),
    ("C1_megapose_official", "C1 MegaPose", "C-finetune"),
    ("C2_gigapose_official", "C2 GigaPose", "C-finetune"),
    ("C3_foundationpose_official", "C3 FoundationPose", "C-finetune"),
    ("C1_megapose_zeroshot", "C1 MegaPose (zs)", "C-zero-shot"),
    ("C2_gigapose_zeroshot", "C2 GigaPose (zs)", "C-zero-shot"),
    ("C3_foundationpose_zeroshot", "C3 FoundationPose (zs)", "C-zero-shot"),
]
EDGES = get_edge_pairs()


def _load_pairs(path, human_matches=None, is_b=False):
    """Return arrays (l_pred, l_gt) for a model dump.

    `human_matches`: optional dict[scene_id] -> set[gt_obj_id] for filtering (407 set).
    `is_b`: if True, recompute l_pred from `kp3d_raw` 24-edge mean (raw path).
    """
    d = json.load(open(path))
    pairs = d["pairs"] if "pairs" in d else d["records"]
    lp, lg = [], []
    for p in pairs:
        if human_matches is not None:
            sid = p.get("scene_id")
            oid = p.get("gt_obj_id", p.get("obj_id"))
            if sid is None or oid is None or oid not in human_matches.get(sid, set()):
                continue
        gl = p.get("gt_edge_length")
        if gl is None or gl <= 0:
            continue
        if is_b and "kp3d_raw" in p:
            kp = np.asarray(p["kp3d_raw"], dtype=np.float64)
            e = compute_edge_lengths_3d(kp[None], EDGES)[0]
            l = float(e.mean())
        else:
            l = float(p["pred_edge_mean"])
        lp.append(l)
        lg.append(gl)
    return np.array(lp, dtype=np.float64), np.array(lg, dtype=np.float64)


def _load_records_gtcrop(path, is_b=False, visibility_filter=None, scene_meta_cache=None):
    """Set A-1 (GT-crop) loader. records[].pred_verts / gt_verts (no Mask R-CNN; uses GT bbox crops).
    Edge length = 24-edge mean from vertices. B-series uses kp3d_raw if present (raw path).

    visibility_filter: optional callable(visibility: float) -> bool. If provided, instances
        whose visibility does not satisfy the predicate are skipped (Set A-2).
    scene_meta_cache: optional dict {sid: {oid: meta}} to avoid re-reading metadata across models.
    """
    d = json.load(open(path))
    recs = d["records"]
    lp, lg = [], []
    for r in recs:
        pv = np.asarray(r["pred_verts"], dtype=np.float64)
        gv = np.asarray(r["gt_verts"], dtype=np.float64)
        if pv.shape != (14, 3) or gv.shape != (14, 3):
            continue
        if visibility_filter is not None:
            sid, oid = r["scene_id"], r["obj_id"]
            if scene_meta_cache is not None and sid not in scene_meta_cache:
                meta_path = os.path.join(DATASET_DIR, f"render_{sid}_metadata.json")
                scene_meta_cache[sid] = {o["id"]: o for o in json.load(open(meta_path))}
            scene_meta = scene_meta_cache[sid] if scene_meta_cache is not None else \
                {o["id"]: o for o in json.load(open(os.path.join(DATASET_DIR, f"render_{sid}_metadata.json")))}
            obj_meta = scene_meta.get(oid, {})
            vis = float(obj_meta.get("visibility", 1.0))
            if not visibility_filter(vis):
                continue
        if is_b and "kp3d_raw" in r:
            kp = np.asarray(r["kp3d_raw"], dtype=np.float64)
            l_pred = float(compute_edge_lengths_3d(kp[None], EDGES)[0].mean())
        else:
            l_pred = float(compute_edge_lengths_3d(pv[None], EDGES)[0].mean())
        l_gt = float(compute_edge_lengths_3d(gv[None], EDGES)[0].mean())
        if l_gt <= 0:
            continue
        lp.append(l_pred)
        lg.append(l_gt)
    return np.array(lp, dtype=np.float64), np.array(lg, dtype=np.float64)


def _hungarian_human_matches(scene_ids):
    matches = {}
    for sid in scene_ids:
        pred = json.load(open(os.path.join(VA_PRED_DIR, f"sem_{sid:05d}.json")))
        gt = json.load(open(os.path.join(VA_GT_DIR, f"sem_{sid:05d}_metadata.json")))
        pc = np.array([r["center"][:2] for r in pred["rhombics"]], dtype=np.float64)
        gc = np.array([o["center"][:2] for o in gt], dtype=np.float64)
        gids = [o["id"] for o in gt]
        if len(pc) == 0 or len(gc) == 0:
            matches[sid] = set()
            continue
        cost = np.linalg.norm(pc[:, None] - gc[None, :], axis=-1)
        _, col_ind = linear_sum_assignment(cost)
        matches[sid] = {gids[c] for c in col_ind}
    return matches


def _human_visibility_lookup(scene_ids):
    """{(sid, oid): visibility} from per-scene metadata."""
    lookup = {}
    for sid in scene_ids:
        gt = json.load(open(os.path.join(VA_GT_DIR, f"sem_{sid:05d}_metadata.json")))
        for o in gt:
            lookup[(sid, o["id"])] = float(o.get("visibility", 1.0))
    return lookup


# Per-instance inference time (ms/instance) — mean and std measured per-instance
# (batch=1, cuda.synchronize) on the missing_human split (239 instances, after warmup).
# Source: `results/per_model_inference_time_ms.json` (regenerated by benchmark_inference_time.py)
def _load_inference_time_table():
    """Returns dict: {key: {mean_ms, std_ms} or None for human}."""
    p = os.path.join(RESULTS_DIR, "per_model_inference_time_ms.json")
    raw_text = open(p).read().replace("NaN", "null") if os.path.exists(p) else "{}"
    try:
        raw = json.loads(raw_text)
    except json.JSONDecodeError:
        raw = {}
    out = {}
    for k in ["A1_gdrnet","A2_sc6d","A3_hccepose","B1_rede","B2_uni6d","B3_ffb6d",
             "C1_megapose_official","C2_gigapose_official","C3_foundationpose_official",
             "C1_megapose_zeroshot","C2_gigapose_zeroshot","C3_foundationpose_zeroshot"]:
        if k in raw and isinstance(raw[k], dict) and "mean_ms" in raw[k]:
            out[k] = {"mean_ms": raw[k]["mean_ms"], "std_ms": raw[k].get("std_ms")}
    out["vertex_annotator"] = None
    return out


INFERENCE_TIME_MS = _load_inference_time_table()


def _vertex_annotator_pairs(scene_ids):
    """Return (l_pred, l_gt) for human vertex_annotator on aligned 407 pairs."""
    lp, lg = [], []
    for sid in scene_ids:
        pred = json.load(open(os.path.join(VA_PRED_DIR, f"sem_{sid:05d}.json")))
        gt = json.load(open(os.path.join(VA_GT_DIR, f"sem_{sid:05d}_metadata.json")))
        pc = np.array([r["center"][:2] for r in pred["rhombics"]], dtype=np.float64)
        plens = np.array([r["length_px"] for r in pred["rhombics"]], dtype=np.float64)
        gc = np.array([o["center"][:2] for o in gt], dtype=np.float64)
        glens = np.array([o["edge_length"] for o in gt], dtype=np.float64)
        if len(pc) == 0 or len(gc) == 0:
            continue
        cost = np.linalg.norm(pc[:, None] - gc[None, :], axis=-1)
        ri, ci = linear_sum_assignment(cost)
        lp.extend(plens[ri].tolist())
        lg.extend(glens[ci].tolist())
    return np.array(lp, dtype=np.float64), np.array(lg, dtype=np.float64)


def metrics(lp, lg):
    delta = lp - lg
    rel = delta / lg
    ratio2 = (lp / lg) ** 2 - 1.0
    abs_delta = np.abs(delta)
    abs_rel_pct = np.abs(rel) * 100.0
    abs_ratio2_pct = np.abs(ratio2) * 100.0
    return {
        "n": int(len(lp)),
        # MAE / MAPE / D-MAE: ± std of |error| (consistent with mean of |·|)
        "edge_mae": float(np.mean(abs_delta)),
        "edge_mae_std": float(np.std(abs_delta)),
        "edge_mape_pct": float(np.mean(abs_rel_pct)),
        "edge_mape_pct_std": float(np.std(abs_rel_pct)),
        "d_mae_pct": float(np.mean(abs_ratio2_pct)),
        "d_mae_pct_std": float(np.std(abs_ratio2_pct)),
        # RMSE / rel-RMSE / D-RMSE: ± std of signed error (= "error spread", companion to RMSE)
        "edge_rmse": float(np.sqrt(np.mean(delta ** 2))),
        "edge_rmse_std": float(np.std(delta)),
        "edge_rel_rmse_pct": float(np.sqrt(np.mean(rel ** 2)) * 100.0),
        "edge_rel_rmse_pct_std": float(np.std(rel) * 100.0),
        "d_rmse_pct": float(np.sqrt(np.mean(ratio2 ** 2)) * 100.0),
        "d_rmse_pct_std": float(np.std(ratio2) * 100.0),
    }


def build_500_scene():
    out = {}
    for key, pretty, cat in MODELS:
        path = os.path.join(PER_MODEL_DIR, f"{key}_maskrcnn_v3_dump.json")
        is_b = cat == "B"
        lp, lg = _load_pairs(path, human_matches=None, is_b=is_b)
        m = metrics(lp, lg)
        m["model"] = pretty
        m["cat"] = cat
        m["key"] = key
        out[key] = m
        print(f"[500-MaskRCNN] {pretty:30s} n={m['n']:5d}  rmse_abs={m['edge_rmse']:.4f}px  "
              f"rel_rmse={m['edge_rel_rmse_pct']:.3f}%  D_rmse={m['d_rmse_pct']:.3f}%")
    return out


def build_500_scene_gtcrop():
    """Set A-1: 500-scene with GT-crop based inference (no Mask R-CNN)."""
    out = {}
    for key, pretty, cat in MODELS:
        path = os.path.join(PER_MODEL_DIR, f"{key}_dump.json")
        is_b = cat == "B"
        lp, lg = _load_records_gtcrop(path, is_b=is_b)
        m = metrics(lp, lg)
        m["model"] = pretty
        m["cat"] = cat
        m["key"] = key
        out[key] = m
        print(f"[500-GTcrop      ] {pretty:30s} n={m['n']:5d}  rmse_abs={m['edge_rmse']:.4f}px  "
              f"rel_rmse={m['edge_rel_rmse_pct']:.3f}%  D_rmse={m['d_rmse_pct']:.3f}%")
    return out


def build_500_scene_gtcrop_filtered():
    """Set A-2: 500-scene GT-crop + fully-visible filter (visibility >= SET_A2_MIN_VISIBILITY)."""
    out = {}
    scene_meta_cache = {}
    visibility_filter = lambda v: v >= SET_A2_MIN_VISIBILITY
    for key, pretty, cat in MODELS:
        path = os.path.join(PER_MODEL_DIR, f"{key}_dump.json")
        is_b = cat == "B"
        lp, lg = _load_records_gtcrop(path, is_b=is_b,
                                       visibility_filter=visibility_filter,
                                       scene_meta_cache=scene_meta_cache)
        m = metrics(lp, lg)
        m["model"] = pretty
        m["cat"] = cat
        m["key"] = key
        out[key] = m
        print(f"[500-GTcrop+vis>={SET_A2_MIN_VISIBILITY}] {pretty:30s} n={m['n']:5d}  rmse_abs={m['edge_rmse']:.4f}px  "
              f"rel_rmse={m['edge_rel_rmse_pct']:.3f}%  D_rmse={m['d_rmse_pct']:.3f}%")
    return out


def build_407():
    aligned = json.load(open(ALIGNED_JSON))
    scene_ids = aligned["scene_ids"]
    human_matches = _hungarian_human_matches(scene_ids)
    out = {}
    for key, pretty, cat in MODELS:
        path = os.path.join(PER_MODEL_DIR, f"{key}_vertex_20img_dump.json")
        is_b = cat == "B"
        lp, lg = _load_pairs(path, human_matches=human_matches, is_b=is_b)
        m = metrics(lp, lg)
        m["model"] = pretty; m["cat"] = cat; m["key"] = key
        out[key] = m
        print(f"[B   ] {pretty:30s} n={m['n']:3d}  rmse_abs={m['edge_rmse']:.4f}px  "
              f"rel_rmse={m['edge_rel_rmse_pct']:.3f}%  D_rmse={m['d_rmse_pct']:.3f}%")
    # vertex_annotator (human)
    lp, lg = _vertex_annotator_pairs(scene_ids)
    m = metrics(lp, lg)
    m["model"] = "vertex_annotator (human)"; m["cat"] = "human"; m["key"] = "vertex_annotator"
    out["vertex_annotator"] = m
    print(f"[B   ] {m['model']:30s} n={m['n']:3d}  rmse_abs={m['edge_rmse']:.4f}px  "
          f"rel_rmse={m['edge_rel_rmse_pct']:.3f}%  D_rmse={m['d_rmse_pct']:.3f}%")
    return out


def _load_pairs_filtered(path, human_matches, vis_lookup, is_b, min_vis):
    """Mask R-CNN dump 에서 human_matches + visibility 필터 후 (l_pred, l_gt) 추출."""
    d = json.load(open(path))
    pairs = d["pairs"] if "pairs" in d else d["records"]
    lp, lg = [], []
    for p in pairs:
        sid = p.get("scene_id")
        oid = p.get("gt_obj_id", p.get("obj_id"))
        if sid is None or oid is None: continue
        if oid not in human_matches.get(sid, set()): continue
        if min_vis is not None:
            v = vis_lookup.get((sid, oid), 1.0)
            if v < min_vis: continue
        gl = p.get("gt_edge_length")
        if gl is None or gl <= 0: continue
        if is_b and "kp3d_raw" in p:
            kp = np.asarray(p["kp3d_raw"], dtype=np.float64)
            l = float(compute_edge_lengths_3d(kp[None], EDGES)[0].mean())
        else:
            l = float(p["pred_edge_mean"])
        lp.append(l); lg.append(gl)
    return np.array(lp, dtype=np.float64), np.array(lg, dtype=np.float64)


def _process_records(recs, human_matches, vis_lookup, is_b, min_vis, seen):
    """Filter and accumulate (l_pred, l_gt) from a record list, tracking already-seen (sid, oid)."""
    lp, lg = [], []
    for r in recs:
        sid, oid = r.get("scene_id"), r.get("obj_id")
        if sid is None or oid is None: continue
        if oid not in human_matches.get(sid, set()): continue
        if (sid, oid) in seen: continue   # avoid double-counting if both dumps overlap
        if min_vis is not None:
            v = vis_lookup.get((sid, oid), 1.0)
            if v < min_vis: continue
        pv = np.asarray(r["pred_verts"], dtype=np.float64)
        gv = np.asarray(r["gt_verts"], dtype=np.float64)
        if pv.shape != (14, 3) or gv.shape != (14, 3): continue
        if is_b and "kp3d_raw" in r:
            kp = np.asarray(r["kp3d_raw"], dtype=np.float64)
            l_pred = float(compute_edge_lengths_3d(kp[None], EDGES)[0].mean())
        else:
            l_pred = float(compute_edge_lengths_3d(pv[None], EDGES)[0].mean())
        l_gt = float(compute_edge_lengths_3d(gv[None], EDGES)[0].mean())
        if l_gt <= 0: continue
        lp.append(l_pred); lg.append(l_gt)
        seen.add((sid, oid))
    return lp, lg


def _load_records_gtcrop_filtered(path_main, path_extra, human_matches, vis_lookup,
                                   is_b, min_vis):
    """GT-crop dump 두 개를 합쳐 human_matches + visibility 필터 후 추출.

    path_main:  `_dump.json` (test split, 168 of 407)
    path_extra: `_human407_missing_dump.json` (12 missing scenes, 239 of 407)
    합치면 전체 407 cover.
    """
    seen: set[tuple] = set()
    lp_all, lg_all = [], []
    for path in (path_main, path_extra):
        if not os.path.exists(path): continue
        d = json.load(open(path))
        recs = d.get("records", d.get("pairs", []))
        lp, lg = _process_records(recs, human_matches, vis_lookup, is_b, min_vis, seen)
        lp_all.extend(lp); lg_all.extend(lg)
    return np.array(lp_all, dtype=np.float64), np.array(lg_all, dtype=np.float64)


def build_407_variant(human_matches, vis_lookup, *, gt_crop, min_vis, label):
    """B / B-1 / B-2 / B-3 variant builder.

    gt_crop=False, min_vis=None     → Set B   (Mask R-CNN, no filter)
    gt_crop=True,  min_vis=None     → Set B-1 (GT crop, no filter)
    gt_crop=False, min_vis=1.0      → Set B-2 (Mask R-CNN + vis=1.0)
    gt_crop=True,  min_vis=1.0      → Set B-3 (GT crop + vis=1.0)
    """
    out = {}
    for key, pretty, cat in MODELS:
        is_b = cat == "B"
        if gt_crop:
            path_main  = os.path.join(PER_MODEL_DIR, f"{key}_dump.json")
            path_extra = os.path.join(PER_MODEL_DIR, f"{key}_human407_missing_dump.json")
            lp, lg = _load_records_gtcrop_filtered(path_main, path_extra, human_matches,
                                                    vis_lookup, is_b, min_vis)
        else:
            path = os.path.join(PER_MODEL_DIR, f"{key}_vertex_20img_dump.json")
            lp, lg = _load_pairs_filtered(path, human_matches, vis_lookup, is_b, min_vis)
        m = metrics(lp, lg)
        m["model"] = pretty; m["cat"] = cat; m["key"] = key
        out[key] = m
        print(f"[{label:5s}] {pretty:30s} n={m['n']:3d}  rmse_abs={m['edge_rmse']:.4f}px  "
              f"rel_rmse={m['edge_rel_rmse_pct']:.3f}%  D_rmse={m['d_rmse_pct']:.3f}%")
    return out


def render_table(metrics_dict, *, sort_key, header_n, include_time=False, fixed_human=None):
    """fixed_human: dict from a reference Set B (407) — if given, vertex_annotator row is replaced
    with these fixed values regardless of the current set's filter (per user request)."""
    rows_dict = dict(metrics_dict)
    if fixed_human is not None and "vertex_annotator" in fixed_human:
        rows_dict["vertex_annotator"] = fixed_human["vertex_annotator"]
    rows = sorted(rows_dict.values(), key=lambda r: r[sort_key])

    cols = ["Rank", "Model", "Cat", "n"]
    if include_time:
        cols.append("Time [ms/inst]")
    cols += ["Edge MAE [px]", "Edge RMSE [px]", "Edge MAPE [%]",
             "Edge rel RMSE [%]", "D / P MAE [%]", "D / P RMSE [%]"]
    lines = ["| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"] * len(cols)) + "|"]

    for i, r in enumerate(rows, 1):
        cells = [str(i), r["model"], r["cat"], str(r["n"])]
        if include_time:
            t = INFERENCE_TIME_MS.get(r["key"])
            if t is None:
                cells.append("—")
            else:
                std = t.get("std_ms")
                if std is None or (isinstance(std, float) and (std != std)):  # nan check
                    cells.append(f"{t['mean_ms']:.4f} ± —")
                else:
                    cells.append(f"{t['mean_ms']:.4f} ± {std:.4f}")
        cells += [
            f"{r['edge_mae']:.4f} ± {r['edge_mae_std']:.4f}",
            f"{r['edge_rmse']:.4f} ± {r['edge_rmse_std']:.4f}",
            f"{r['edge_mape_pct']:.4f} ± {r['edge_mape_pct_std']:.4f}",
            f"{r['edge_rel_rmse_pct']:.4f} ± {r['edge_rel_rmse_pct_std']:.4f}",
            f"{r['d_mae_pct']:.4f} ± {r['d_mae_pct_std']:.4f}",
            f"{r['d_rmse_pct']:.4f} ± {r['d_rmse_pct_std']:.4f}",
        ]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main():
    print("=== Set A: 500-scene (Mask R-CNN-detected, no human) ===")
    m500 = build_500_scene()
    print()
    print("=== Set A-1: 500-scene (GT-crop based, no Mask R-CNN, no human) ===")
    m500_gt = build_500_scene_gtcrop()
    print()
    print(f"=== Set A-2: 500-scene (GT-crop + visibility >= {SET_A2_MIN_VISIBILITY}, no human) ===")
    m500_gt_vis = build_500_scene_gtcrop_filtered()
    print()
    print("=== Set B: 407-instance (human-aligned, Mask R-CNN-detected) ===")
    m407 = build_407()

    aligned = json.load(open(ALIGNED_JSON))
    scene_ids = aligned["scene_ids"]
    human_matches = _hungarian_human_matches(scene_ids)
    vis_lookup = _human_visibility_lookup(scene_ids)
    print()
    print("=== Set B-1: 407 (GT-crop based, no Mask R-CNN) ===")
    m407_b1 = build_407_variant(human_matches, vis_lookup, gt_crop=True,  min_vis=None, label="B-1")
    print()
    print("=== Set B-2: 407 ∩ visibility = 1.0 (Mask R-CNN-detected) ===")
    m407_b2 = build_407_variant(human_matches, vis_lookup, gt_crop=False, min_vis=1.0,  label="B-2")
    print()
    print("=== Set B-3: 407 ∩ visibility = 1.0 + GT-crop ===")
    m407_b3 = build_407_variant(human_matches, vis_lookup, gt_crop=True,  min_vis=1.0,  label="B-3")

    md_path = os.path.join(RESULTS_DIR, "edge_d_p_ranking_two_sets.md")
    json_path = os.path.join(RESULTS_DIR, "edge_d_p_ranking_two_sets.json")

    n_500 = next(iter(m500.values()))["n"]
    n_500_gt = next(iter(m500_gt.values()))["n"]
    n_500_gt_vis = next(iter(m500_gt_vis.values()))["n"]
    n_407 = m407["vertex_annotator"]["n"]
    n_b1 = next(iter(m407_b1.values()))["n"]
    n_b2 = next(iter(m407_b2.values()))["n"]
    n_b3 = next(iter(m407_b3.values()))["n"]
    body = []
    body.append("# Edge / Diffusivity / Permeability — 500-scene & 407-instance Ranking")
    body.append("")
    body.append(f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    body.append(f"- **Set A**: 500 test scenes — **Mask R-CNN-detected** pairs (no human), n={n_500}")
    body.append(f"- **Set A-1**: 500 test scenes — **GT-crop based** inference (no Mask R-CNN, no human), n={n_500_gt}")
    body.append(f"- **Set A-2**: 500 test scenes — **GT-crop + {SET_A2_LABEL}** filter (no Mask R-CNN, no occlusion, no human), n={n_500_gt_vis}")
    body.append(f"- **Set B**: 407 human-aligned instances (20 SEM-styled scenes, "
                f"Mask R-CNN + Hungarian matched to vertex_annotator clicks)")
    body.append(f"- Object diameter: d = 6.928 px (orthographic camera, world unit = pixel)")
    body.append("- B-series: raw kp3d 24-edge mean (raw path, no Procrustes) in Set A / A-1 / A-2 / B.")
    body.append("- Set A vs Set A-1: 동일 모델, 동일 500 scene 위에서 **detection 단계 유무만 차이**.")
    body.append("  Set A 는 Mask R-CNN bbox 로 인스턴스를 검출 후 GT 와 Hungarian 매칭 (실제 운영 파이프라인).")
    body.append("  Set A-1 은 GT bbox 를 직접 crop 으로 사용 → detection 노이즈 제거된 \"upper bound\" 측정.")
    body.append("- Set A-1 vs Set A-2: 동일 GT-crop 위에서 **occlusion 영향만 차이**.")
    body.append(f"  Set A-2 는 Set A-1 인스턴스 중 visibility = 1.0 (완전히 보이는 인스턴스) 만 유지.")
    body.append("  Set A-1 → Set A-2 갭이 \"가려짐 영향\" 의 직접적 정량 측정.")
    body.append("- Sort: ascending **Edge rel RMSE [%]** (lower = better).")
    body.append("- ± 표기 = per-instance scalar의 표준편차 (training run variance가 아니라 instance-level 변동).")
    body.append("  - MAE / MAPE / D · P MAE: $\\sigma(|\\Delta l|)$, $\\sigma(|\\Delta l/l_{gt}|)$, $\\sigma(|(l_p/l_g)^2-1|)$ — mean of |·| 과 동일한 underlying scalar")
    body.append("  - RMSE / rel-RMSE / D · P RMSE: $\\sigma(\\Delta l)$, $\\sigma(\\Delta l/l_{gt})$, $\\sigma((l_p/l_g)^2-1)$ — signed error의 spread (RMSE companion: $\\text{RMSE}^2 = \\sigma^2 + \\text{bias}^2$)")
    body.append("")
    body.append("## Notation")
    body.append("- $l_{pred,i}$ : i-th instance predicted edge length [px]")
    body.append("- $l_{gt,i}$ : i-th instance GT edge length [px]")
    body.append("- $\\Delta l_i = l_{pred,i} - l_{gt,i}$")
    body.append("- N : number of instances in the set (500-scene → 11,375 ; 407 human-aligned → 407)")
    body.append("")
    body.append("## 사용 수식")
    body.append("")
    body.append("### (1) Edge MAE — 절대 (px)")
    body.append("$$\\text{MAE}_{\\text{edge}} = \\frac{1}{N}\\sum_{i=1}^{N}|l_{pred,i} - l_{gt,i}| \\quad [\\text{px}]$$")
    body.append("")
    body.append("### (2) Edge RMSE — 절대 (px)")
    body.append("$$\\text{RMSE}_{\\text{edge,abs}} = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}(l_{pred,i} - l_{gt,i})^2} \\quad [\\text{px}]$$")
    body.append("")
    body.append("### (3) Edge MAPE — 상대 평균 (%)")
    body.append("$$\\text{MAPE}_{\\text{edge}} = \\frac{1}{N}\\sum_{i=1}^{N}\\left|\\frac{l_{pred,i} - l_{gt,i}}{l_{gt,i}}\\right| \\times 100 \\quad [\\%]$$")
    body.append("")
    body.append("### (4) Edge RMSE — 상대 (%)")
    body.append("$$\\text{RMSE}_{\\text{edge,rel}} = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}\\left(\\frac{l_{pred,i} - l_{gt,i}}{l_{gt,i}}\\right)^2} \\times 100 \\quad [\\%]$$")
    body.append("")
    body.append("객체 크기로 정규화 → 작은 객체 에러를 더 무겁게 가중.")
    body.append("")
    body.append("### (5) Diffusivity / Permeability MAE & RMSE — 정확 (%)")
    body.append("From `fomulation.md`: $D \\propto l^2$, $P = D \\cdot S$ with constant $S$ → $\\Delta P/P = \\Delta D/D$. 따라서:")
    body.append("$$\\frac{\\Delta D_i}{D_{gt,i}} = \\left(\\frac{l_{pred,i}}{l_{gt,i}}\\right)^2 - 1$$")
    body.append("$$\\text{MAE}_{D} = \\text{MAE}_{P} = \\frac{1}{N}\\sum_{i=1}^{N}\\left|\\left(\\frac{l_{pred,i}}{l_{gt,i}}\\right)^2 - 1\\right| \\times 100 \\quad [\\%]$$")
    body.append("$$\\text{RMSE}_{D} = \\text{RMSE}_{P} = \\sqrt{\\frac{1}{N}\\sum_{i=1}^{N}\\left(\\left(\\frac{l_{pred,i}}{l_{gt,i}}\\right)^2 - 1\\right)^2} \\times 100 \\quad [\\%]$$")
    body.append("")
    body.append("First-order Taylor: $\\text{RMSE}_{D} \\approx 2 \\cdot \\text{RMSE}_{\\text{edge,rel}}$. 본 분석은 정확 식 사용.")
    body.append("")
    body.append(f"## Set A — 500 test scene (n={n_500}, Mask R-CNN-detected, no human)")
    body.append("")
    body.append(render_table(m500, sort_key="edge_rel_rmse_pct", header_n=n_500))
    body.append("")
    body.append(f"## Set A-1 — 500 test scene (n={n_500_gt}, GT-crop based, no Mask R-CNN, no human)")
    body.append("")
    body.append("Detection 단계 노이즈를 제거하여 \"pose 추정 자체의 한계\" 만 측정한 결과. "
                "Set A 와의 차이 = Mask R-CNN 의 영향.")
    body.append("")
    body.append(render_table(m500_gt, sort_key="edge_rel_rmse_pct", header_n=n_500_gt))
    body.append("")
    body.append(f"## Set A-2 — 500 test scene (n={n_500_gt_vis}, GT-crop + {SET_A2_LABEL})")
    body.append("")
    body.append("GT-crop 사용 (detection 영향 제거) + visibility = 1.0 인 인스턴스만 유지 (occlusion 영향 제거). "
                "두 가지 외부 noise source 가 모두 제거된 \"clean\" 평가 — pose 추정 모델의 **순수 capacity 상한**. "
                f"전체 11,379 instance 중 {n_500_gt_vis} 개 (~{100*n_500_gt_vis/n_500_gt:.0f}%) 가 fully visible 조건 만족.")
    body.append("")
    body.append(render_table(m500_gt_vis, sort_key="edge_rel_rmse_pct", header_n=n_500_gt_vis))
    body.append("")
    body.append(f"## Set B — 407 instance (human-aligned, Mask R-CNN-detected)")
    body.append("")
    body.append("운영 파이프라인 (Mask R-CNN detection + Hungarian human matching). 407 paired instance.")
    body.append("")
    body.append("**Time [ms/inst]** = 모델별 per-instance pose inference 시간 (mean ± std). "
                "측정 방법: batch=1 forward pass × 234 instance (warmup 5 후), cuda.synchronize 기반. "
                "Mask R-CNN detection 시간 미포함, 모델만의 forward pass 시간. "
                "human 은 click 작업이라 별도 측정 안 함. "
                "C3 zero-shot 은 monolithic refinement loop 라 std `—` 표기 (mean 만 aggregate 측정).")
    body.append("")
    body.append(render_table(m407, sort_key="edge_rel_rmse_pct", header_n=n_407, include_time=True))
    body.append("")
    body.append(f"## Set B-1 — 407 instance (GT-crop based, no Mask R-CNN)")
    body.append("")
    body.append(f"같은 407 instance 위에서 detection 단계만 GT bbox 로 변경. Set B 와의 차이 = **Mask R-CNN detection 영향**.")
    body.append(f"")
    body.append(f"**Inference 방식**: 8 scene (test split 포함) 은 `_dump.json`, 12 missing scene (별도 inference) 은 `_human407_missing_dump.json` 에서 가져와 합침. **n = {n_b1} (전체 407 cover)**.")
    body.append(f"")
    body.append(f"human 은 fixed reference 로 407 그대로 표시.")
    body.append("")
    body.append(render_table(m407_b1, sort_key="edge_rel_rmse_pct", header_n=n_b1,
                             include_time=True, fixed_human=m407))
    body.append("")
    body.append(f"## Set B-2 — 407 instance ∩ visibility = 1.0 (Mask R-CNN-detected)")
    body.append("")
    body.append(f"Set B 인스턴스 중 visibility = 1.0 (완전히 보이는) 만 유지. 가려짐 영향 제거. "
                f"n = {n_b2} (407 중 ~{100*n_b2/n_407:.0f}%). human 은 fixed reference 로 407 그대로.")
    body.append("")
    body.append(render_table(m407_b2, sort_key="edge_rel_rmse_pct", header_n=n_b2,
                             include_time=True, fixed_human=m407))
    body.append("")
    body.append(f"## Set B-3 — 407 instance ∩ visibility = 1.0 + GT-crop (둘 다 보장)")
    body.append("")
    body.append(f"Set B 인스턴스 중 visibility = 1.0 + GT-crop. detection / occlusion 양쪽 noise 제거. "
                f"이는 Set B 의 \"이상적 상한\" 평가 (pose 모델 자체 한계만 측정).")
    body.append(f"")
    body.append(f"**Inference 방식**: B-1 과 동일하게 8 + 12 scene 합쳐 전체 407 위에서 visibility=1.0 필터 적용. "
                f"**n = {n_b3} (407 중 ~{100*n_b3/n_407:.0f}%, fully visible 조건 만족)**. "
                f"human 은 fixed reference 로 407 그대로.")
    body.append("")
    body.append(render_table(m407_b3, sort_key="edge_rel_rmse_pct", header_n=n_b3,
                             include_time=True, fixed_human=m407))
    body.append("")
    body.append("### Set B / B-1 / B-2 / B-3 비교 — noise source 별 영향 분해")
    body.append("")
    body.append("동일 407 human-aligned 기반에서 단계적으로 noise source 를 제거했을 때의 rel-RMSE 변화. "
                "human 은 fixed (407 instance reference) 로 간주.")
    body.append("")
    body.append("| Model | Cat | Set B | Set B-1 (GT crop) | Set B-2 (vis=1.0) | Set B-3 (둘 다) | Δ MaskRCNN (B−B-1) | Δ occlusion (B-B-2) |")
    body.append("|---|---|---|---|---|---|---|---|")
    for key, _, _ in MODELS:
        b  = m407[key]["edge_rel_rmse_pct"]
        b1 = m407_b1[key]["edge_rel_rmse_pct"]
        b2 = m407_b2[key]["edge_rel_rmse_pct"]
        b3 = m407_b3[key]["edge_rel_rmse_pct"]
        d_seg = b  - b1
        d_occ = b  - b2
        body.append(f"| {m407[key]['model']} | {m407[key]['cat']} | "
                    f"{b:.4f} | {b1:.4f} | {b2:.4f} | {b3:.4f} | "
                    f"{d_seg:+.4f} | {d_occ:+.4f} |")
    h = m407["vertex_annotator"]
    body.append(f"| vertex_annotator (human) | human | {h['edge_rel_rmse_pct']:.4f} | "
                f"{h['edge_rel_rmse_pct']:.4f} (fixed) | {h['edge_rel_rmse_pct']:.4f} (fixed) | "
                f"{h['edge_rel_rmse_pct']:.4f} (fixed) | — | — |")
    body.append("")
    body.append("## Set A vs Set A-1 vs Set A-2 — noise source 별 영향 분해")
    body.append("")
    body.append("같은 500 scene, 같은 모델 위에서 단계적으로 noise source 를 제거했을 때의 메트릭 변화.")
    body.append("- Δ (A − A-1) = Mask R-CNN detection 노이즈가 기여하는 양")
    body.append("- Δ (A-1 − A-2) = 가려짐 (occlusion) 이 기여하는 양 (fully-visible filter 효과)")
    body.append("- Set A-2 = pose 추정 자체의 noise (모든 외부 noise 제거 후 남은 양)")
    body.append("")
    body.append("**rel-RMSE 분해 [%]**")
    body.append("")
    body.append("| Model | Cat | Set A | Set A-1 | Set A-2 | Δ MaskRCNN (A − A-1) | Δ occlusion (A-1 − A-2) | A-2 = pose noise alone |")
    body.append("|---|---|---|---|---|---|---|---|")
    for key, _, _ in MODELS:
        a = m500[key]; a1 = m500_gt[key]; a2 = m500_gt_vis[key]
        d_seg = a["edge_rel_rmse_pct"]  - a1["edge_rel_rmse_pct"]
        d_occ = a1["edge_rel_rmse_pct"] - a2["edge_rel_rmse_pct"]
        body.append(f"| {a['model']} | {a['cat']} | {a['edge_rel_rmse_pct']:.4f} | "
                    f"{a1['edge_rel_rmse_pct']:.4f} | {a2['edge_rel_rmse_pct']:.4f} | "
                    f"{d_seg:+.4f} | {d_occ:+.4f} | {a2['edge_rel_rmse_pct']:.4f} |")
    body.append("")
    body.append("**D / P RMSE 분해 [%]**")
    body.append("")
    body.append("| Model | Cat | Set A | Set A-1 | Set A-2 | Δ MaskRCNN | Δ occlusion |")
    body.append("|---|---|---|---|---|---|---|")
    for key, _, _ in MODELS:
        a = m500[key]; a1 = m500_gt[key]; a2 = m500_gt_vis[key]
        d_seg = a["d_rmse_pct"]  - a1["d_rmse_pct"]
        d_occ = a1["d_rmse_pct"] - a2["d_rmse_pct"]
        body.append(f"| {a['model']} | {a['cat']} | {a['d_rmse_pct']:.4f} | "
                    f"{a1['d_rmse_pct']:.4f} | {a2['d_rmse_pct']:.4f} | "
                    f"{d_seg:+.4f} | {d_occ:+.4f} |")
    body.append("")
    body.append("## 산출 파일")
    body.append("- `results/edge_d_p_ranking_two_sets.md` (이 파일)")
    body.append("- `results/edge_d_p_ranking_two_sets.json` (raw metric values)")
    md = "\n".join(body) + "\n"

    with open(md_path, "w") as f:
        f.write(md)
    with open(json_path, "w") as f:
        json.dump(
            {
                "date": datetime.now().isoformat(),
                "set_500_scene_maskrcnn": m500,
                "set_500_scene_gtcrop":   m500_gt,
                "set_500_scene_gtcrop_fully_visible": m500_gt_vis,
                "set_500_scene_gtcrop_min_visibility": SET_A2_MIN_VISIBILITY,
                "set_407_human_aligned":  m407,
                "set_407_b1_gtcrop":      m407_b1,
                "set_407_b2_visfilter":   m407_b2,
                "set_407_b3_gtcrop_visfilter": m407_b3,
                "inference_time_ms": INFERENCE_TIME_MS,
            },
            f,
            indent=2,
        )
    print(f"\nwrote {md_path}")
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
