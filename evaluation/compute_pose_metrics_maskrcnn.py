"""maskrcnn_v3 dump → ADD-S, R err, t err 추가 계산하여 v3.1-complete summary 생성.

dump JSON에서 R_pred/t_pred/s_pred + gt_verts 사용:
- R_gt/t_gt/s_gt는 Umeyama로 gt_verts 에서 복원
- ADD-S, R err, t err은 lineup geometry의 sym_aware_* 함수 사용 (compute_metrics와 동일)

새 inference 없음. visualizations/ 절대 건드리지 않음.
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, "<PROJECT_ROOT>/baselines")
from common import geometry as geo

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from edge_utils import get_canonical_vertices


REPO = "<PROJECT_ROOT>"
DUMP_DIR = os.path.join(REPO, "edge_length_experiment", "results", "per_model")
RESULTS = os.path.join(REPO, "edge_length_experiment", "results")
DIAMETER = 6.928203230275509

PRETTY = {
    "A1_gdrnet": "A1 GDR-Net", "A2_sc6d": "A2 SC6D", "A3_hccepose": "A3 HccePose",
    "B1_rede": "B1 REDE", "B2_uni6d": "B2 Uni6D", "B3_ffb6d": "B3 FFB6D",
    "C1_megapose_official": "C1 MegaPose",
    "C2_gigapose_official": "C2 GigaPose",
    "C3_foundationpose_official": "C3 FoundationPose",
    "C1_megapose_zeroshot": "C1 MegaPose (zero-shot)",
    "C2_gigapose_zeroshot": "C2 GigaPose (zero-shot)",
    "C3_foundationpose_zeroshot": "C3 FoundationPose (zero-shot)",
}

# GT-mask reference (BASELINES_SUMMARY.md 본문값)
GT_MASK_REF = {
    "A1_gdrnet":   {"add_s": 109.92, "r_err": 19.24, "t_err": 105.99, "mape": 9.55},
    "A2_sc6d":     {"add_s": 110.02, "r_err": 18.25, "t_err": 106.59, "mape": 7.28},
    "A3_hccepose": {"add_s": 140.06, "r_err":  6.95, "t_err": 139.05, "mape": 5.43},
    "B1_rede":     {"add_s": 108.23, "r_err": 14.72, "t_err": 105.60, "mape": 7.84},
    "B2_uni6d":    {"add_s": 110.17, "r_err": 22.03, "t_err": 105.74, "mape": 10.54},
    "B3_ffb6d":    {"add_s": 108.42, "r_err": 16.03, "t_err": 105.37, "mape": 8.63},
    "C1_megapose_official":        {"add_s": 103.22, "r_err": 11.49, "t_err": 101.21, "mape": 6.52},
    "C2_gigapose_official":        {"add_s": 100.81, "r_err": 12.94, "t_err":  98.38, "mape": 5.38},
    "C3_foundationpose_official":  {"add_s": 103.19, "r_err":  8.88, "t_err": 101.90, "mape": 5.14},
    "C1_megapose_zeroshot":       {"add_s": 157.99, "r_err": 39.01, "t_err": 149.95, "mape": 15.39},
    "C2_gigapose_zeroshot":       {"add_s": 157.99, "r_err": 39.01, "t_err": 149.95, "mape": 15.39},
    "C3_foundationpose_zeroshot": {"add_s": 158.42, "r_err": 40.99, "t_err": 149.95, "mape": 15.39},
}


def load_dump(name):
    p = os.path.join(DUMP_DIR, f"{name}_maskrcnn_v3_dump.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))


def edge_metrics_from_pairs(pairs):
    abs_err = np.array([p["abs_err"] for p in pairs])
    sq_err = np.array([p["squared_err"] for p in pairs])
    rel_err_pct = np.abs(np.array([p["rel_err"] for p in pairs])) * 100.0
    return {"n": len(pairs), "mae": float(abs_err.mean()),
            "rmse": float(np.sqrt(sq_err.mean())),
            "mape": float(rel_err_pct.mean())}


def b_raw_mpjpe_pck_from_pairs(pairs, canon, O24, diameter):
    if not pairs or "kp3d_raw" not in pairs[0]:
        return {}
    th_05 = 0.5 * diameter
    th_10 = 1.0 * diameter
    mpjpes, pcks_05, pcks_10 = [], [], []
    for p in pairs:
        kp3d = np.asarray(p["kp3d_raw"], dtype=np.float64)
        gt = np.asarray(p["gt_verts"], dtype=np.float64)
        R_gt, t_gt, s_gt = geo.umeyama_np(canon, gt)
        best_mean = np.inf
        best_dists = None
        for G in O24:
            sym_gt = s_gt * (canon @ G.T) @ R_gt.T + t_gt
            dists = np.linalg.norm(kp3d - sym_gt, axis=1)
            m = float(dists.mean())
            if m < best_mean:
                best_mean = m; best_dists = dists
        mpjpes.append(best_mean)
        pcks_05.append(float((best_dists < th_05).mean()))
        pcks_10.append(float((best_dists < th_10).mean()))
    return {"mpjpe": float(np.mean(mpjpes)),
            "pck_05d": float(np.mean(pcks_05) * 100.0),
            "pck_10d": float(np.mean(pcks_10) * 100.0)}


def pose_metrics_from_pairs(pairs, canon):
    """sym-aware ADD-S, R err, t err (compute_metrics와 동일 정의)."""
    add_s_list, r_err_list, t_err_list = [], [], []
    for p in pairs:
        R_p = np.asarray(p["R_pred"], dtype=np.float64)
        t_p = np.asarray(p["t_pred"], dtype=np.float64)
        s_p = float(p["s_pred"])
        pred_verts = s_p * canon @ R_p.T + t_p
        gt_verts = np.asarray(p["gt_verts"], dtype=np.float64)
        # GT R/t/s 복원
        R_gt, t_gt, s_gt = geo.umeyama_np(canon, gt_verts)
        # ADD-S sym-aware
        add = geo.sym_aware_vertex_error(pred_verts, gt_verts, canon,
                                           R_gt, t_gt, float(s_gt))
        add_s_list.append(float(add["add_s_min"]))
        # R err sym-aware
        r_err = geo.sym_aware_rot_error_deg(R_p, R_gt)
        r_err_list.append(float(r_err))
        # t err — non-sym, world frame
        t_err_list.append(float(np.linalg.norm(t_p - t_gt)))
    return {
        "add_s": float(np.mean(add_s_list)),
        "r_err": float(np.mean(r_err_list)),
        "t_err": float(np.mean(t_err_list)),
    }


def main():
    canon = get_canonical_vertices().astype(np.float64)
    O24 = geo.O24_rotations()

    CAT_A    = ["A1_gdrnet", "A2_sc6d", "A3_hccepose"]
    CAT_B    = ["B1_rede", "B2_uni6d", "B3_ffb6d"]
    CAT_C_FT = ["C1_megapose_official", "C2_gigapose_official", "C3_foundationpose_official"]
    CAT_C_ZS = ["C1_megapose_zeroshot", "C2_gigapose_zeroshot", "C3_foundationpose_zeroshot"]

    metrics = {}
    for name in CAT_A + CAT_B + CAT_C_FT + CAT_C_ZS:
        d = load_dump(name)
        if d is None:
            print(f"[v3.1] missing {name}")
            continue
        em = edge_metrics_from_pairs(d["pairs"])
        print(f"[v3.1] {name} pose metrics ...", flush=True)
        t0 = time.time()
        pm = pose_metrics_from_pairs(d["pairs"], canon)
        em.update(pm)
        if name in CAT_B:
            mp = b_raw_mpjpe_pck_from_pairs(d["pairs"], canon, O24, DIAMETER)
            em.update(mp)
        print(f"  done {time.time()-t0:.1f}s  add_s={pm['add_s']:.4f}  r_err={pm['r_err']:.4f}  t_err={pm['t_err']:.4f}")
        metrics[name] = em

    def sort_table(model_list):
        rows = []
        for name in model_list:
            if name not in metrics:
                continue
            rows.append({"name": name, "label": PRETTY[name], **metrics[name]})
        rows.sort(key=lambda r: r["mape"])
        return rows

    table_a    = sort_table(CAT_A)
    table_b    = sort_table(CAT_B)
    table_c_ft = sort_table(CAT_C_FT)
    table_c_zs = sort_table(CAT_C_ZS)
    all_rows = table_a + table_b + table_c_ft + table_c_zs

    md = []
    md.append("# Edge Length — Mask R-CNN end-to-end (test split 500 scenes) v3.1-complete (with pose metrics)")
    md.append("")
    md.append("## Setup")
    md.append(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("- 입력: **500 styled images** (`dataset_v6/render_NNNNN_styled.png`, test split 18000~19999)")
    md.append("- Detection: **Mask R-CNN seg_exp07**")
    md.append("- BASELINES inference: forward_eval (finetune) / paper checkpoint + template-match or register() (zero-shot)")
    md.append(f"- Object diameter d = {DIAMETER:.4f} px")
    md.append("- Sort: ascending **MAPE** (lower = better), Bold = best in each table column")
    md.append("- **v3.1**: Pose metrics (ADD-S, R err, t err) 추가 — 모두 **sym-aware** (O_24 group)")
    md.append("- All metrics computed on identical 11,375 matched detection-GT pairs")
    md.append("")
    md.append("## Notation / Units")
    md.append("- ADD-S, t err, MAE, RMSE, MPJPE: pixels (px) — orthographic 카메라, world unit = px")
    md.append("- R err: degrees (°)")
    md.append("- MAPE, PCK: percentage (%)")
    md.append("")

    def emit_table(title, header_cols, rows, has_keypoint=False):
        md.append(f"## {title}")
        md.append("")
        md.append("| " + " | ".join(header_cols) + " |")
        md.append("|" + "|".join(["---"] * len(header_cols)) + "|")
        if not rows:
            md.append("| — |" * len(header_cols))
            md.append("")
            return
        # bold each numeric column's best
        keys_for_min = ["add_s", "r_err", "t_err", "mae", "rmse", "mape"]
        if has_keypoint:
            keys_for_min += ["mpjpe"]
            keys_for_max = ["pck_05d", "pck_10d"]
        else:
            keys_for_max = []
        bests = {k: min(r[k] for r in rows if k in r) for k in keys_for_min if any(k in r for r in rows)}
        bests_max = {k: max(r[k] for r in rows if k in r) for k in keys_for_max if any(k in r for r in rows)}

        def cell(r, k, fmt):
            v = r.get(k)
            if v is None:
                return "—"
            tag = ""
            if k in bests and v == bests[k]:
                tag = "**"
            elif k in bests_max and v == bests_max[k]:
                tag = "**"
            return f"{tag}{fmt(v)}{tag}"

        for i, r in enumerate(rows, 1):
            cells = [str(i), r["label"]]
            cells.append(cell(r, "add_s", lambda v: f"{v:.4f}"))
            cells.append(cell(r, "r_err", lambda v: f"{v:.4f}"))
            cells.append(cell(r, "t_err", lambda v: f"{v:.4f}"))
            if has_keypoint:
                cells.append(cell(r, "mpjpe", lambda v: f"{v:.4f}"))
                cells.append(cell(r, "pck_05d", lambda v: f"{v:.4f}"))
                cells.append(cell(r, "pck_10d", lambda v: f"{v:.4f}"))
            cells.append(cell(r, "mae", lambda v: f"{v:.4f}"))
            cells.append(cell(r, "rmse", lambda v: f"{v:.4f}"))
            cells.append(cell(r, "mape", lambda v: f"{v:.2f}"))
            md.append("| " + " | ".join(cells) + " |")
        md.append("")

    emit_table("Table 1 — Category A (RGB regression)",
               ["Rank", "Model", "ADD-S (px)", "R err (°)", "t err (px)",
                "MAE (px)", "RMSE (px)", "MAPE (%)"],
               table_a)
    emit_table("Table 2 — Category B (3D keypoint regression, raw kp3d)",
               ["Rank", "Model", "ADD-S (px)", "R err (°)", "t err (px)",
                "MPJPE (px)", "PCK@0.5d (%)", "PCK@1.0d (%)",
                "MAE (px)", "RMSE (px)", "MAPE (%)"],
               table_b, has_keypoint=True)
    emit_table("Table 3 — Category C — Foundation pose (finetune)",
               ["Rank", "Model", "ADD-S (px)", "R err (°)", "t err (px)",
                "MAE (px)", "RMSE (px)", "MAPE (%)"],
               table_c_ft)
    emit_table("Table 4 — Category C — Foundation pose (zero-shot)",
               ["Rank", "Model", "ADD-S (px)", "R err (°)", "t err (px)",
                "MAE (px)", "RMSE (px)", "MAPE (%)"],
               table_c_zs)

    # 핵심 관찰
    md.append("## 핵심 관찰 (v3.1 신규 분석)")
    md.append("")
    if all_rows:
        best_mape = min(all_rows, key=lambda r: r["mape"])
        md.append(f"1. **MAPE 1위**: {best_mape['label']} ({best_mape['mape']:.2f}%, ADD-S {best_mape['add_s']:.2f}, R err {best_mape['r_err']:.2f}°)")
        best_add = min(all_rows, key=lambda r: r["add_s"])
        md.append(f"2. **ADD-S 1위**: {best_add['label']} ({best_add['add_s']:.4f} px, MAPE {best_add['mape']:.2f}%)")
        best_r = min(all_rows, key=lambda r: r["r_err"])
        md.append(f"3. **R err 1위**: {best_r['label']} ({best_r['r_err']:.2f}°, MAPE {best_r['mape']:.2f}%)")
        best_t = min(all_rows, key=lambda r: r["t_err"])
        md.append(f"4. **t err 1위**: {best_t['label']} ({best_t['t_err']:.2f} px)")
        md.append("")

    # GT-mask vs Mask R-CNN pose metrics 비교
    md.append("### GT-mask vs Mask R-CNN — Pose metrics 비교")
    md.append("")
    md.append("| Model | ADD-S Δ (RCNN−GT) | R err Δ (°) | t err Δ (px) | MAPE Δ (%p) |")
    md.append("|---|---|---|---|---|")
    for r in sorted(all_rows, key=lambda x: x["mape"]):
        gt = GT_MASK_REF.get(r["name"])
        if gt is None:
            continue
        d_add = r["add_s"] - gt["add_s"]
        d_r = r["r_err"] - gt["r_err"]
        d_t = r["t_err"] - gt["t_err"]
        d_m = r["mape"] - gt["mape"]
        s = lambda v: ("+" if v >= 0 else "") + f"{v:.2f}"
        md.append(f"| {r['label']} | {s(d_add)} | {s(d_r)} | {s(d_t)} | {s(d_m)} |")
    md.append("")
    md.append("- 12 모델 평균 ΔADD-S = "
              + f"+{np.mean([r['add_s'] - GT_MASK_REF[r['name']]['add_s'] for r in all_rows if r['name'] in GT_MASK_REF]):.2f}")
    md.append("- 12 모델 평균 ΔR err = "
              + f"+{np.mean([r['r_err'] - GT_MASK_REF[r['name']]['r_err'] for r in all_rows if r['name'] in GT_MASK_REF]):.2f}°")
    md.append("- 12 모델 평균 Δt err = "
              + f"+{np.mean([r['t_err'] - GT_MASK_REF[r['name']]['t_err'] for r in all_rows if r['name'] in GT_MASK_REF]):.2f} px")
    md.append("")

    # Edge MAPE vs ADD-S vs R err ranking 일치도
    md.append("### Ranking — Edge MAPE / ADD-S / R err 사이 비교")
    md.append("")
    md.append("| Rank | by MAPE | by ADD-S | by R err |")
    md.append("|---|---|---|---|")
    by_mape = sorted(all_rows, key=lambda r: r["mape"])
    by_add = sorted(all_rows, key=lambda r: r["add_s"])
    by_rerr = sorted(all_rows, key=lambda r: r["r_err"])
    for i in range(len(by_mape)):
        md.append(f"| {i+1} | {by_mape[i]['label']} | {by_add[i]['label']} | {by_rerr[i]['label']} |")
    md.append("")

    md.append("### Edge MAPE vs Pose metrics 상관 (Pearson r)")
    md.append("")
    if len(all_rows) >= 3:
        x_mape = np.array([r["mape"] for r in all_rows])
        x_add = np.array([r["add_s"] for r in all_rows])
        x_r = np.array([r["r_err"] for r in all_rows])
        x_t = np.array([r["t_err"] for r in all_rows])
        def corr(a, b):
            return float(np.corrcoef(a, b)[0, 1])
        md.append(f"- MAPE ↔ ADD-S: r = {corr(x_mape, x_add):.3f}")
        md.append(f"- MAPE ↔ R err: r = {corr(x_mape, x_r):.3f}")
        md.append(f"- MAPE ↔ t err: r = {corr(x_mape, x_t):.3f}")
    md.append("")

    md.append("## Caption (LaTeX)")
    md.append("")
    md.append("> All metrics computed on identical 11,375 matched detection-GT pairs from Mask R-CNN end-to-end pipeline.")
    md.append("> Pipeline: Mask R-CNN (seg_exp07_maskrcnn_v6_improved) detects rhombic instances → Hungarian center matching to GT → BASELINES forward_eval (finetune) or paper-pretrained inference (zero-shot) → R/t/s + 14 vertex output.")
    md.append("> Pose metrics: ADD-S in pixels (px), R err in degrees (°), t err in pixels (px), all symmetry-aware (O_24 group).")
    md.append("> Edge length metrics computed from 24 unique edges of rhombic dodecahedron (all equal by isohedral property).")
    md.append("> Units: ADD-S, t err, MAE, RMSE, MPJPE in pixels (px); R err in degrees (°); MAPE and PCK in percentage (%).")
    md.append("")
    md.append("## 산출 파일")
    md.append("- `results/summary_maskrcnn_v3.1_complete.{md,csv,json}` (이 라운드)")
    md.append("- 기존 `summary_maskrcnn_v3_complete.*` + 모든 dump JSON 보존")
    md.append("- visualizations/ 디렉토리 절대 건드리지 않음")

    md_path = os.path.join(RESULTS, "summary_maskrcnn_v3.1_complete.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {md_path}")

    csv_path = os.path.join(RESULTS, "summary_maskrcnn_v3.1_complete.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Category", "Rank", "Model",
                    "ADD-S_px", "R_err_deg", "t_err_px",
                    "MPJPE_px", "PCK_05d_pct", "PCK_10d_pct",
                    "MAE_px", "RMSE_px", "MAPE_pct"])
        for cat_name, tbl in [("A", table_a), ("B", table_b),
                                ("C-finetune", table_c_ft), ("C-zero-shot", table_c_zs)]:
            for i, r in enumerate(tbl, 1):
                w.writerow([
                    cat_name, i, r["label"],
                    f"{r.get('add_s', ''):.4f}" if "add_s" in r else "",
                    f"{r.get('r_err', ''):.4f}" if "r_err" in r else "",
                    f"{r.get('t_err', ''):.4f}" if "t_err" in r else "",
                    f"{r.get('mpjpe', ''):.4f}" if "mpjpe" in r else "",
                    f"{r.get('pck_05d', ''):.4f}" if "pck_05d" in r else "",
                    f"{r.get('pck_10d', ''):.4f}" if "pck_10d" in r else "",
                    f"{r['mae']:.4f}", f"{r['rmse']:.4f}", f"{r['mape']:.4f}",
                ])
    print(f"wrote {csv_path}")

    json_path = os.path.join(RESULTS, "summary_maskrcnn_v3.1_complete.json")
    with open(json_path, "w") as f:
        json.dump({
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "diameter_px": DIAMETER,
            "tables": {
                "A": table_a, "B": table_b,
                "C_finetune": table_c_ft, "C_zero_shot": table_c_zs,
            },
            "gt_mask_reference": GT_MASK_REF,
        }, f, indent=2)
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
