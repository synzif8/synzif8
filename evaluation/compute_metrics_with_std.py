"""두 데이터셋의 metrics 재계산 + std 추가 (소수점 4자리).

입력:
- Synthetic 500: per_model/{name}_maskrcnn_v3_dump.json (12 모델)
- SEM-style 20:  per_model/{name}_vertex_20img_dump.json (12 모델)
                + length_annotations + sem_images_json (vertex_annotator human)

출력: results/benchmark_results_v3.2_with_std.{md,csv,json}
"""

from __future__ import annotations

import csv
import json
import os
import sys
import time

import numpy as np
from scipy.optimize import linear_sum_assignment


REPO = "<PROJECT_ROOT>"
DUMP_DIR = os.path.join(REPO, "edge_length_experiment", "results", "per_model")
RESULTS = os.path.join(REPO, "edge_length_experiment", "results")

VA_PRED_DIR = "<PROJECT_ROOT>/annotation_tool/length_annotations"
VA_GT_DIR = "<PROJECT_ROOT>/annotation_tool/sem_images_json"

DIAMETER = 6.928203230275509

PRETTY = {
    "A1_gdrnet": "A1 GDR-Net", "A2_sc6d": "A2 SC6D", "A3_hccepose": "A3 HccePose",
    "B1_rede": "B1 REDE", "B2_uni6d": "B2 Uni6D", "B3_ffb6d": "B3 FFB6D",
    "C1_megapose_official": "C1 MegaPose",
    "C2_gigapose_official": "C2 GigaPose",
    "C3_foundationpose_official": "C3 FoundationPose",
    "C1_megapose_zeroshot": "C1 MegaPose (zs)",
    "C2_gigapose_zeroshot": "C2 GigaPose (zs)",
    "C3_foundationpose_zeroshot": "C3 FoundationPose (zs)",
    "vertex_annotator": "vertex_annotator (human)",
}
CATEGORY = {
    **{m: "A" for m in ["A1_gdrnet", "A2_sc6d", "A3_hccepose"]},
    **{m: "B" for m in ["B1_rede", "B2_uni6d", "B3_ffb6d"]},
    **{m: "C-finetune" for m in ["C1_megapose_official", "C2_gigapose_official", "C3_foundationpose_official"]},
    **{m: "C-zero-shot" for m in ["C1_megapose_zeroshot", "C2_gigapose_zeroshot", "C3_foundationpose_zeroshot"]},
    "vertex_annotator": "human",
}

ALL_20_SCENES = [18003, 18019, 18022, 18040, 18041, 18045, 18052, 18055, 18058, 18060,
                 18063, 18064, 18067, 18073, 18076, 18081, 18095, 18097, 18098, 18100]


def metrics_with_std(pairs):
    """per-pair raw → MAE_mean ± std, RMSE, MAPE_mean ± std (4 decimals)."""
    abs_err = np.array([p["abs_err"] for p in pairs])
    sq_err = np.array([p["squared_err"] for p in pairs])
    rel_pct = np.abs(np.array([p["rel_err"] for p in pairs])) * 100.0
    return {
        "n": len(pairs),
        "mae_mean": float(abs_err.mean()),
        "mae_std":  float(abs_err.std(ddof=0)),
        "rmse":     float(np.sqrt(sq_err.mean())),
        "mape_mean": float(rel_pct.mean()),
        "mape_std":  float(rel_pct.std(ddof=0)),
    }


def load_dump_pairs(name, suffix):
    p = os.path.join(DUMP_DIR, f"{name}_{suffix}_dump.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["pairs"]


def vertex_annotator_pairs():
    """length_annotations + sem_images_json metadata 에서 per-pair 재계산 (Hungarian match)."""
    pairs_all = []
    for sid in ALL_20_SCENES:
        pred_path = os.path.join(VA_PRED_DIR, f"sem_{sid:04d}.json")
        gt_path = os.path.join(VA_GT_DIR, f"sem_{sid:04d}_metadata.json")
        if not os.path.exists(pred_path) or not os.path.exists(gt_path):
            continue
        pred = json.load(open(pred_path))
        gt = json.load(open(gt_path))
        pred_centers = np.array([r["center"][:2] for r in pred["rhombics"]])
        pred_lens = np.array([r["length_px"] for r in pred["rhombics"]])
        gt_centers = np.array([o["center"][:2] for o in gt])
        gt_lens = np.array([o["edge_length"] for o in gt])
        if len(pred_centers) == 0 or len(gt_centers) == 0:
            continue
        cost = np.linalg.norm(pred_centers[:, None, :] - gt_centers[None, :, :], axis=-1)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            pl = float(pred_lens[r]); gl = float(gt_lens[c])
            pairs_all.append({
                "scene_id": int(sid),
                "abs_err": abs(pl - gl),
                "rel_err": (pl - gl) / max(abs(gl), 1e-9),
                "squared_err": (pl - gl) ** 2,
            })
    return pairs_all


def main():
    print("[v3.2] computing Synthetic 500 metrics ...")
    syn_results = {}
    for name in PRETTY:
        if name == "vertex_annotator":
            continue
        pairs = load_dump_pairs(name, "maskrcnn_v3")
        if pairs is None:
            print(f"  missing synthetic 500 dump: {name}")
            continue
        syn_results[name] = metrics_with_std(pairs)

    print("[v3.2] computing SEM-style 20 metrics ...")
    sem_results = {}
    for name in PRETTY:
        if name == "vertex_annotator":
            pairs = vertex_annotator_pairs()
            if pairs:
                sem_results[name] = metrics_with_std(pairs)
        else:
            pairs = load_dump_pairs(name, "vertex_20img")
            if pairs is None:
                print(f"  missing SEM 20 dump: {name}")
                continue
            sem_results[name] = metrics_with_std(pairs)

    def fmt_pm(m, s): return f"{m:.4f} ± {s:.4f}"
    def fmt(v): return f"{v:.4f}"

    md = []
    md.append("# Benchmark Results with ± std (소수점 4자리)")
    md.append("")
    md.append("## Setup")
    md.append(f"- Date: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    md.append("- **Synthetic 500**: Mask R-CNN end-to-end on 500-scene test split (n=11,375 matched pairs)")
    md.append("  - dump: `results/per_model/{name}_maskrcnn_v3_dump.json` × 12")
    md.append("- **SEM-style 20**: vertex_annotator + 12 BASELINES (n=426 for AI, n=407 for human)")
    md.append("  - AI dump: `results/per_model/{name}_vertex_20img_dump.json` × 12")
    md.append("  - Human pairs: re-matched from `length_annotations/` + `sem_images_json/` (Hungarian by 2D center)")
    md.append("- All ± values: standard deviation across instances (per-pair variation)")
    md.append("- Sort: ascending **MAPE** (lower = better), Bold = best")
    md.append("- All values rounded to **4 decimals**")
    md.append("")
    md.append("## 메트릭 형식")
    md.append("- **MAE**: mean ± std")
    md.append("- **RMSE**: 단일값")
    md.append("- **MAPE**: mean ± std")
    md.append("")

    # Synthetic 500 table
    rows_syn = []
    for name in PRETTY:
        if name not in syn_results:
            continue
        r = syn_results[name]
        rows_syn.append({"name": name, "label": PRETTY[name],
                         "category": CATEGORY[name], **r})
    rows_syn.sort(key=lambda r: r["mape_mean"])

    md.append("## Synthetic 500 통합 표 (12 모델)")
    md.append("")
    md.append("| Rank | Category | Model | n | MAE [px] | RMSE [px] | MAPE [%] |")
    md.append("|---|---|---|---|---|---|---|")
    if rows_syn:
        best_mape = min(r["mape_mean"] for r in rows_syn)
        for i, r in enumerate(rows_syn, 1):
            b = "**" if r["mape_mean"] == best_mape else ""
            md.append(f"| {i} | {r['category']} | {b}{r['label']}{b} | {r['n']} | "
                      f"{b}{fmt_pm(r['mae_mean'], r['mae_std'])}{b} | "
                      f"{b}{fmt(r['rmse'])}{b} | "
                      f"{b}{fmt_pm(r['mape_mean'], r['mape_std'])}{b} |")
    md.append("")

    # SEM-style 20 table
    rows_sem = []
    for name in PRETTY:
        if name not in sem_results:
            continue
        r = sem_results[name]
        rows_sem.append({"name": name, "label": PRETTY[name],
                         "category": CATEGORY[name], **r})
    rows_sem.sort(key=lambda r: r["mape_mean"])

    md.append("## SEM-style 20 통합 표 (13 entries: 12 모델 + 사람)")
    md.append("")
    md.append("| Rank | Category | Model | n | MAE [px] | RMSE [px] | MAPE [%] |")
    md.append("|---|---|---|---|---|---|---|")
    if rows_sem:
        best_mape = min(r["mape_mean"] for r in rows_sem)
        for i, r in enumerate(rows_sem, 1):
            b = "**" if r["mape_mean"] == best_mape else ""
            md.append(f"| {i} | {r['category']} | {b}{r['label']}{b} | {r['n']} | "
                      f"{b}{fmt_pm(r['mae_mean'], r['mae_std'])}{b} | "
                      f"{b}{fmt(r['rmse'])}{b} | "
                      f"{b}{fmt_pm(r['mape_mean'], r['mape_std'])}{b} |")
    md.append("")

    # Aggregated paper table
    md.append("## Aggregated 표 (paper LaTeX용)")
    md.append("")
    md.append("| Method | Cat | Synthetic RMSE | Synthetic MAPE | SEM RMSE | SEM MAPE |")
    md.append("|---|---|---|---|---|---|")
    paper_models = ["A1_gdrnet", "A2_sc6d", "A3_hccepose",
                     "B1_rede", "B2_uni6d", "B3_ffb6d",
                     "C1_megapose_official", "C2_gigapose_official", "C3_foundationpose_official",
                     "C1_megapose_zeroshot", "C2_gigapose_zeroshot", "C3_foundationpose_zeroshot",
                     "vertex_annotator"]
    for name in paper_models:
        s = syn_results.get(name)
        e = sem_results.get(name)
        cat = CATEGORY.get(name, "?")
        s_rmse = fmt(s["rmse"]) if s else "—"
        s_map = fmt_pm(s["mape_mean"], s["mape_std"]) if s else "—"
        e_rmse = fmt(e["rmse"]) if e else "—"
        e_map = fmt_pm(e["mape_mean"], e["mape_std"]) if e else "—"
        md.append(f"| {PRETTY.get(name, name)} | {cat} | {s_rmse} | {s_map} | {e_rmse} | {e_map} |")
    md.append("")

    # 핵심 관찰
    md.append("## 핵심 관찰")
    md.append("")
    if rows_syn:
        best = rows_syn[0]
        md.append(f"1. **Synthetic 500 MAPE 1위**: {best['label']} ({fmt_pm(best['mape_mean'], best['mape_std'])}%)")
    if rows_sem:
        best_e = rows_sem[0]
        md.append(f"2. **SEM-style 20 MAPE 1위**: {best_e['label']} ({fmt_pm(best_e['mape_mean'], best_e['mape_std'])}%)")
    md.append("")
    md.append("3. **Synthetic ↔ SEM-style MAPE 비교** (12 모델):")
    md.append("")
    md.append("| Model | Synthetic MAPE | SEM MAPE | Δ (SEM−Syn) |")
    md.append("|---|---|---|---|")
    for name in paper_models:
        if name == "vertex_annotator":
            continue
        s = syn_results.get(name); e = sem_results.get(name)
        if s and e:
            d = e["mape_mean"] - s["mape_mean"]
            sign = "+" if d >= 0 else ""
            md.append(f"| {PRETTY[name]} | {s['mape_mean']:.4f} | {e['mape_mean']:.4f} | {sign}{d:.4f} |")
    md.append("")

    md.append("## Caption (LaTeX)")
    md.append("")
    md.append("> Edge length benchmark on (i) Synthetic 500-scene test split (n=11,375 matched detection-GT pairs, Mask R-CNN end-to-end);")
    md.append("> (ii) SEM-style 20 images (n=426 for AI, n=407 for human, all images in test pool [18000, 20000) unseen by training).")
    md.append("> All ± values are standard deviation across per-instance errors. Units: MAE, RMSE in pixels (px); MAPE in percentage (%).")
    md.append(f"> Object diameter d = {DIAMETER:.4f} px.")
    md.append("")
    md.append("## 산출 파일")
    md.append("- `results/benchmark_results_v3.2_with_std.{md,csv,json}` (이 라운드)")
    md.append("- 입력 dump (read-only): `per_model/{name}_maskrcnn_v3_dump.json`, `per_model/{name}_vertex_20img_dump.json`")
    md.append("- 기존 v3, v3.1-complete 등 모두 보존")

    md_path = os.path.join(RESULTS, "benchmark_results_v3.2_with_std.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    print(f"wrote {md_path}")

    # CSV (combined)
    csv_path = os.path.join(RESULTS, "benchmark_results_v3.2_with_std.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Dataset", "Category", "Model", "n_pairs",
                    "MAE_mean_px", "MAE_std_px", "RMSE_px",
                    "MAPE_mean_pct", "MAPE_std_pct"])
        for name in paper_models:
            for ds_label, ds in [("Synthetic_500", syn_results), ("SEM_20", sem_results)]:
                r = ds.get(name)
                if r is None:
                    continue
                w.writerow([
                    ds_label, CATEGORY.get(name, "?"), PRETTY[name], r["n"],
                    f"{r['mae_mean']:.4f}", f"{r['mae_std']:.4f}", f"{r['rmse']:.4f}",
                    f"{r['mape_mean']:.4f}", f"{r['mape_std']:.4f}",
                ])
    print(f"wrote {csv_path}")

    # JSON
    json_path = os.path.join(RESULTS, "benchmark_results_v3.2_with_std.json")
    with open(json_path, "w") as f:
        json.dump({
            "date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "diameter_px": DIAMETER,
            "synthetic_500": syn_results,
            "sem_style_20": sem_results,
        }, f, indent=2)
    print(f"wrote {json_path}")


if __name__ == "__main__":
    main()
