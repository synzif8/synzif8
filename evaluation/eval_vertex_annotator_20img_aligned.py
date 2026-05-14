"""20-image vertex_annotator-aligned BASELINES comparison.

Idea
----
Original `vertex_annotator_eval_20img` had n_pairs mismatch (vertex_annotator 407,
BASELINES 426). The human missed 19 GT objects. This script identifies which
specific (scene_id, gt_obj_id) pairs the human matched (via the same Hungarian
center matching used in the original eval) and filters all 12 BASELINES dumps to
that exact subset, producing apples-to-apples 407-vs-407 comparison.

Inputs
------
- <PROJECT_ROOT>/annotation_tool/length_annotations/sem_NNNNN.json
  (rhombics[].center → human pred centers)
- <PROJECT_ROOT>/annotation_tool/sem_images_json/sem_NNNNN_metadata.json
  (GT objects, [].id and [].center)
- edge_length_experiment/results/per_model/{model}_vertex_20img_dump.json × 12
  (pairs[].scene_id, .gt_obj_id, .abs_err, .squared_err, .rel_err, .gt_edge_length)
- edge_length_experiment/results/vertex_annotator_eval_20img.json
  (master JSON, vertex_annotator per-scene already on the matched set)

Outputs
-------
- results/vertex_annotator_eval_20img_aligned.md
- results/vertex_annotator_eval_20img_aligned.json
- results/vertex_annotator_eval_20img_aligned.csv
"""
import json
import os
import sys
from datetime import datetime

import numpy as np
from scipy.optimize import linear_sum_assignment


VA_PRED_DIR = "<PROJECT_ROOT>/annotation_tool/length_annotations"
VA_GT_DIR = "<PROJECT_ROOT>/annotation_tool/sem_images_json"
RESULTS_DIR = "<PROJECT_ROOT>/evaluation/results"
PER_MODEL_DIR = os.path.join(RESULTS_DIR, "per_model")

MASTER_JSON = os.path.join(RESULTS_DIR, "vertex_annotator_eval_20img.json")

MODELS_ORDER = [
    ("A1_gdrnet", "A"),
    ("A2_sc6d", "A"),
    ("A3_hccepose", "A"),
    ("B1_rede", "B"),
    ("B2_uni6d", "B"),
    ("B3_ffb6d", "B"),
    ("C1_megapose_official", "C-finetune"),
    ("C2_gigapose_official", "C-finetune"),
    ("C3_foundationpose_official", "C-finetune"),
    ("C1_megapose_zeroshot", "C-zero-shot"),
    ("C2_gigapose_zeroshot", "C-zero-shot"),
    ("C3_foundationpose_zeroshot", "C-zero-shot"),
]
PRETTY_NAME = {
    "A1_gdrnet": "A1 GDR-Net",
    "A2_sc6d": "A2 SC6D",
    "A3_hccepose": "A3 HccePose",
    "B1_rede": "B1 REDE",
    "B2_uni6d": "B2 Uni6D",
    "B3_ffb6d": "B3 FFB6D",
    "C1_megapose_official": "C1 MegaPose",
    "C2_gigapose_official": "C2 GigaPose",
    "C3_foundationpose_official": "C3 FoundationPose",
    "C1_megapose_zeroshot": "C1 MegaPose (zs)",
    "C2_gigapose_zeroshot": "C2 GigaPose (zs)",
    "C3_foundationpose_zeroshot": "C3 FoundationPose (zs)",
}


def hungarian_human_matches(scene_ids):
    """Per scene: which gt obj_ids did the human match? Returns dict[sid] -> set[int]."""
    matches = {}
    n_pairs_check = {}
    for sid in scene_ids:
        pred_path = os.path.join(VA_PRED_DIR, f"sem_{sid:05d}.json")
        gt_path = os.path.join(VA_GT_DIR, f"sem_{sid:05d}_metadata.json")
        pred = json.load(open(pred_path))
        gt = json.load(open(gt_path))
        pred_centers = np.array(
            [r["center"][:2] for r in pred["rhombics"]], dtype=np.float64
        )
        gt_centers = np.array([o["center"][:2] for o in gt], dtype=np.float64)
        gt_ids = [o["id"] for o in gt]
        if len(pred_centers) == 0 or len(gt_centers) == 0:
            matches[sid] = set()
            n_pairs_check[sid] = 0
            continue
        cost = np.linalg.norm(
            pred_centers[:, None, :] - gt_centers[None, :, :], axis=-1
        )
        row_ind, col_ind = linear_sum_assignment(cost)
        matched_ids = {gt_ids[c] for c in col_ind}
        matches[sid] = matched_ids
        n_pairs_check[sid] = len(matched_ids)
    return matches, n_pairs_check


def filter_and_aggregate(model_key, human_matches):
    """Filter a model's pairs to human-matched GT and recompute metrics."""
    dump_path = os.path.join(PER_MODEL_DIR, f"{model_key}_vertex_20img_dump.json")
    d = json.load(open(dump_path))
    pairs = d["pairs"]
    per_scene = {}
    all_abs = []
    all_sq = []
    all_rel = []
    all_gt_edge = []
    for p in pairs:
        sid = p["scene_id"]
        oid = p["gt_obj_id"]
        if oid not in human_matches.get(sid, set()):
            continue
        all_abs.append(p["abs_err"])
        all_sq.append(p["squared_err"])
        # Original rel_err = (pred - gt)/gt; we need MAPE = |rel|*100 over pairs
        all_rel.append(abs(p["rel_err"]))
        all_gt_edge.append(p["gt_edge_length"])
        per_scene.setdefault(sid, {"abs": [], "sq": [], "rel": []})
        per_scene[sid]["abs"].append(p["abs_err"])
        per_scene[sid]["sq"].append(p["squared_err"])
        per_scene[sid]["rel"].append(abs(p["rel_err"]))

    def agg(abs_l, sq_l, rel_l):
        n = len(abs_l)
        if n == 0:
            return {"n": 0, "mae": 0.0, "rmse": 0.0, "mape": 0.0}
        return {
            "n": n,
            "mae": float(np.mean(abs_l)),
            "rmse": float(np.sqrt(np.mean(sq_l))),
            "mape": float(np.mean(rel_l) * 100.0),
        }

    out = {
        "aggregate": agg(all_abs, all_sq, all_rel),
        "per_scene": {
            sid: agg(v["abs"], v["sq"], v["rel"])
            for sid, v in per_scene.items()
        },
    }
    return out


def main():
    master = json.load(open(MASTER_JSON))
    scene_ids = master["scene_ids"]

    print(f"[1/3] Hungarian re-matching for {len(scene_ids)} scenes...")
    human_matches, n_pairs_check = hungarian_human_matches(scene_ids)
    total_human_pairs = sum(len(v) for v in human_matches.values())
    print(f"  recovered total pairs: {total_human_pairs}")
    # Validate
    expected = master["models"]["vertex_annotator"]["aggregate"]["n_pairs"]
    assert (
        total_human_pairs == expected
    ), f"Hungarian recovery mismatch: {total_human_pairs} vs master {expected}"
    print(f"  ✓ matches master vertex_annotator n_pairs ({expected})")

    print("[2/3] Filtering 12 BASELINES dumps...")
    aligned_models = {}
    for model_key, _cat in MODELS_ORDER:
        result = filter_and_aggregate(model_key, human_matches)
        aligned_models[model_key] = result
        print(
            f"  {model_key:35s} n={result['aggregate']['n']:3d} "
            f"MAPE={result['aggregate']['mape']:5.2f}%"
        )

    # Add vertex_annotator from master (already aligned)
    va_master = master["models"]["vertex_annotator"]
    aligned_models["vertex_annotator"] = {
        "aggregate": {
            "n": va_master["aggregate"]["n_pairs"],
            "mae": va_master["aggregate"]["mae"],
            "rmse": va_master["aggregate"]["rmse"],
            "mape": va_master["aggregate"]["mape"],
        },
        "per_scene": {
            int(sid): {
                "n": v["n_pairs"],
                "mae": v["mae"],
                "rmse": v["rmse"],
                "mape": v["mape"],
            }
            for sid, v in va_master["per_scene"].items()
        },
    }

    print("[3/3] Writing outputs...")

    # Sort by aggregate MAPE
    rank_keys = list(aligned_models.keys())
    rank_keys.sort(key=lambda k: aligned_models[k]["aggregate"]["mape"])

    cat_lookup = {k: c for k, c in MODELS_ORDER}
    cat_lookup["vertex_annotator"] = "human"
    pretty_lookup = dict(PRETTY_NAME)
    pretty_lookup["vertex_annotator"] = "vertex_annotator (human)"

    # MD output
    md = []
    md.append("# Vertex Annotator + BASELINES Edge Length 평가 (20 scenes, human-aligned)")
    md.append("")
    md.append("## Setup")
    md.append(
        f"- Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    md.append(
        "- 입력: **20장 SEM-style 이미지** "
        "(`<PROJECT_ROOT>/annotation_tool/sem_like_image/sem_NNNNN.png`)"
    )
    md.append(
        "- 평가 대상 GT 객체: **사람이 vertex_annotator로 매칭한 객체만** "
        f"(전체 {sum(len(g['points']) for sid in scene_ids for g in [json.load(open(os.path.join(VA_GT_DIR, f'sem_{sid:05d}_metadata.json')))[0]]) // 14 if False else 426} 중 **{total_human_pairs}** 개; 19 빠짐)"
    )
    md.append(
        "- Detection: **Mask R-CNN seg_exp07_maskrcnn_v6_improved/best_model.pth** (이전과 동일)"
    )
    md.append(
        "- BASELINES inference: 9 finetune + 3 zero-shot (각 모델 dump → human-matched (scene_id, gt_obj_id) 필터)"
    )
    md.append(
        "- vertex_annotator (human): Mask R-CNN seg + 사람이 vertex 2개 클릭 → 픽셀 거리 (master JSON 그대로 사용)"
    )
    md.append("- Object diameter d = 6.9282 px")
    md.append("- Sort: ascending **MAPE** (lower = better), Bold = best")
    md.append("")

    # Aggregate table
    md.append(f"## Aggregate ({total_human_pairs} pairs, all 12 BASELINES + vertex_annotator on identical GT subset)")
    md.append("")
    md.append("| Rank | Model | Cat | n_pairs | MAE [px] | RMSE [px] | MAPE [%] |")
    md.append("|---|---|---|---|---|---|---|")
    for i, k in enumerate(rank_keys, 1):
        a = aligned_models[k]["aggregate"]
        bold = "**" if i == 1 else ""
        md.append(
            f"| {i} | {bold}{pretty_lookup[k]}{bold} | {cat_lookup[k]} | "
            f"{a['n']} | {bold}{a['mae']:.4f}{bold} | {bold}{a['rmse']:.4f}{bold} | "
            f"{bold}{a['mape']:.2f}{bold} |"
        )
    md.append("")

    # Comparison table: original 20-scene vs aligned
    md.append("## Aligned vs Original 20-scene MAPE (Δ = aligned − original)")
    md.append("")
    md.append("| Model | Original n=426 MAPE | Aligned n=407 MAPE | Δ |")
    md.append("|---|---|---|---|")
    orig = master["models"]
    for k in rank_keys:
        if k == "vertex_annotator":
            orig_mape = orig[k]["aggregate"]["mape"]
        else:
            orig_mape = orig[k]["aggregate"]["mape"]
        aligned_mape = aligned_models[k]["aggregate"]["mape"]
        diff = aligned_mape - orig_mape
        md.append(
            f"| {pretty_lookup[k]} | {orig_mape:.2f} | {aligned_mape:.2f} | {diff:+.2f} |"
        )
    md.append("")

    # Per-scene MAPE
    md.append("## Per-scene MAPE (aligned subset only)")
    md.append("")
    header = "| Scene | " + " | ".join(pretty_lookup[k] for k in rank_keys) + " |"
    sep = "|---|" + "|".join("---" for _ in rank_keys) + "|"
    md.append(header)
    md.append(sep)
    for sid in scene_ids:
        row = [str(sid)]
        for k in rank_keys:
            ps = aligned_models[k].get("per_scene", {}).get(sid)
            if ps is None:
                ps = aligned_models[k].get("per_scene", {}).get(int(sid))
            row.append(f"{ps['mape']:.2f}" if ps else "—")
        md.append("| " + " | ".join(row) + " |")
    md.append("")

    # Missed objects breakdown
    md.append("## 사람이 놓친 GT 객체 (per scene)")
    md.append("")
    md.append("| Scene | n_gt | n_human | missed gt_obj_id |")
    md.append("|---|---|---|---|")
    n_missed_total = 0
    for sid in scene_ids:
        gt = json.load(open(os.path.join(VA_GT_DIR, f"sem_{sid:05d}_metadata.json")))
        all_ids = {o["id"] for o in gt}
        matched = human_matches[sid]
        missed = sorted(all_ids - matched)
        n_missed_total += len(missed)
        md.append(
            f"| {sid} | {len(all_ids)} | {len(matched)} | {missed if missed else '—'} |"
        )
    md.append(f"\n**Total missed**: {n_missed_total}")
    md.append("")

    md.append("## 핵심 관찰")
    md.append("")
    aligned_va = aligned_models["vertex_annotator"]["aggregate"]
    aligned_top_basel = next(
        (
            k
            for k in rank_keys
            if k != "vertex_annotator"
        ),
        None,
    )
    md.append(
        f"- **MAPE 1위**: {pretty_lookup[rank_keys[0]]} ({aligned_models[rank_keys[0]]['aggregate']['mape']:.2f}%)"
    )
    md.append(
        f"- vertex_annotator (human) MAPE: {aligned_va['mape']:.2f}% — 동일 set으로 비교 시 BASELINES와 fair compare"
    )
    md.append(
        f"- 19개 GT 객체가 사람에 의해 누락됨 → 이들을 BASELINES에서도 동일하게 제외"
    )
    md.append("")

    md.append("## Caption (LaTeX)")
    md.append("")
    md.append(
        "> Edge length evaluation on 20 SEM-style images, restricted to the 407 GT instances "
        "successfully annotated by the human vertex_annotator. The 19 GT objects missed by the "
        "human are excluded from BASELINES evaluation as well, ensuring identical-instance comparison. "
        "Human-matched GT identities are recovered via Hungarian center assignment between human "
        "vertex clicks and ground-truth instance centers, then propagated to each BASELINES dump's "
        "(scene_id, gt_obj_id) records. Units: MAE, RMSE in pixels; MAPE in %. Object diameter d = 6.928 px."
    )
    md.append("")
    md.append("## 산출 파일")
    md.append("- `results/vertex_annotator_eval_20img_aligned.{md,json,csv}`")
    md.append("")

    md_text = "\n".join(md)
    md_path = os.path.join(RESULTS_DIR, "vertex_annotator_eval_20img_aligned.md")
    with open(md_path, "w") as f:
        f.write(md_text)
    print(f"  ✓ {md_path}")

    # JSON
    json_path = os.path.join(RESULTS_DIR, "vertex_annotator_eval_20img_aligned.json")
    out = {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "n_scenes": len(scene_ids),
        "scene_ids": scene_ids,
        "diameter_px": 6.928203230275509,
        "n_human_pairs": total_human_pairs,
        "human_matches_per_scene": {
            int(sid): sorted(list(v)) for sid, v in human_matches.items()
        },
        "models": aligned_models,
    }
    with open(json_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"  ✓ {json_path}")

    # CSV
    csv_path = os.path.join(RESULTS_DIR, "vertex_annotator_eval_20img_aligned.csv")
    with open(csv_path, "w") as f:
        f.write("rank,model,category,n_pairs,mae,rmse,mape\n")
        for i, k in enumerate(rank_keys, 1):
            a = aligned_models[k]["aggregate"]
            f.write(
                f"{i},{pretty_lookup[k]},{cat_lookup[k]},{a['n']},"
                f"{a['mae']:.6f},{a['rmse']:.6f},{a['mape']:.6f}\n"
            )
    print(f"  ✓ {csv_path}")

    print(f"\nDone. Aligned comparison on {total_human_pairs} pairs.")


if __name__ == "__main__":
    main()
