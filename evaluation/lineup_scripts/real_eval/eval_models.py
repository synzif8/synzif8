"""Evaluate 9 models against the v2 (A+C-only) GT-ID labeling.

For each rhombic and each model M:
  1. Find σ_M ∈ Π_24 minimising 2D Procrustes residual of M's 14-vertex template
     to the AC_only consensus template (i.e., relabel M's own ID convention onto
     the consensus convention).
  2. For each GT click i with assigned consensus-ID j_i, compute model M's
     predicted 2D position as T_M[σ_M[j_i]].
  3. Per-click pixel error = ||click_i - T_M[σ_M[j_i]]||₂.

Crucial: the alignment σ_M is *labelling-only* — we then evaluate M's *raw*
projected pixel positions, no further Procrustes/scale/translation transform.
So the metric reflects M's true sim-to-real pose accuracy, not just shape
similarity.

Outputs:
  runs/full/_real_eval/AC_only/MODEL_EVAL.json    — raw per-model + per-tier
  runs/full/_real_eval/AC_only/MODEL_EVAL.md      — human-readable table
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

import numpy as np

REPO = "<PROJECT_ROOT>/baselines"
sys.path.insert(0, REPO)

from scripts.real_eval.consensus import _PERMS, proc2d

PRED_DIR = os.path.join(REPO, "runs/full/_real_eval/preds")
ASSIGN_DIR = os.path.join(REPO, "runs/full/_real_eval/AC_only/assign")
OUT_JSON = os.path.join(REPO, "runs/full/_real_eval/AC_only/MODEL_EVAL.json")
OUT_MD = os.path.join(REPO, "runs/full/_real_eval/AC_only/MODEL_EVAL.md")

PCK_THR = [3, 5, 7, 10, 15, 20, 30]


def best_sigma(template_M: np.ndarray, consensus: np.ndarray) -> tuple[int, float]:
    best_r = float("inf")
    best_k = 0
    for k, perm in enumerate(_PERMS):
        r, _ = proc2d(template_M[perm], consensus)
        if r < best_r:
            best_r = r
            best_k = k
    return best_k, best_r


def evaluate(threshold_voting_inliers: int = 5,
             threshold_assign_mean: float = 15.0):
    """Tier 1: full set, Tier 2: confident subset (gating: inliers ≥ thr AND assign_mean ≤ thr)."""
    pred_files = {os.path.splitext(os.path.basename(p))[0]: p
                  for p in sorted(glob(os.path.join(PRED_DIR, "*.npz")))}
    assign_files = {os.path.splitext(os.path.basename(p))[0]: p
                    for p in sorted(glob(os.path.join(ASSIGN_DIR, "*.npz")))}

    # Per-model accumulators (full and gated)
    model_names = None
    full_errs: dict[str, list[float]] = {}
    gated_errs: dict[str, list[float]] = {}
    full_sigma_resid: dict[str, list[float]] = {}

    n_total_clicks = 0
    n_gated_clicks = 0
    n_total_rhombics = 0
    n_gated_rhombics = 0

    for stem, pred_p in pred_files.items():
        if stem not in assign_files:
            print(f"[skip] {stem}: no assignment")
            continue
        pred = np.load(pred_p, allow_pickle=True)
        ass = np.load(assign_files[stem], allow_pickle=True)

        verts = pred["verts_2d_full"]      # (N, 9, 14, 2)
        names_full = pred["model_names"].tolist()
        gt_full = pred["gt_full"]          # object array
        consensus = ass["consensus_templates"]  # (N, 14, 2)
        per_rh = ass["per_rhombic"]        # object array of dicts

        if model_names is None:
            model_names = names_full
            for n in model_names:
                full_errs[n] = []
                gated_errs[n] = []
                full_sigma_resid[n] = []

        N, M, _, _ = verts.shape
        for n_idx in range(N):
            cons = consensus[n_idx]                  # (14, 2)
            gt = gt_full[n_idx]                       # (Ng, 2)
            if gt.shape[0] == 0:
                continue
            rec = per_rh[n_idx]
            assigned_ids = np.asarray(rec["assigned_ids"], np.int32)  # (Ng,) consensus convention
            inlier_count = int(rec["inlier_count"])
            assign_mean = float(rec["assign_mean_cost"])
            tier1 = (inlier_count >= threshold_voting_inliers and
                     assign_mean <= threshold_assign_mean)

            n_total_rhombics += 1
            if tier1:
                n_gated_rhombics += 1
            n_total_clicks += gt.shape[0]
            if tier1:
                n_gated_clicks += gt.shape[0]

            for m, mn in enumerate(names_full):
                T_m = verts[n_idx, m]                # (14, 2)
                sig_idx, sig_res = best_sigma(T_m, cons)
                full_sigma_resid[mn].append(sig_res)
                sigma = _PERMS[sig_idx]              # (14,)
                # M's predicted pixel for consensus-ID j is T_m[sigma[j]]
                pred_pts = T_m[sigma[assigned_ids]]  # (Ng, 2)
                errs = np.linalg.norm(gt - pred_pts, axis=1).astype(float)
                full_errs[mn].extend(errs.tolist())
                if tier1:
                    gated_errs[mn].extend(errs.tolist())

    return dict(
        model_names=model_names,
        full_errs=full_errs,
        gated_errs=gated_errs,
        full_sigma_resid=full_sigma_resid,
        n_total_clicks=n_total_clicks,
        n_gated_clicks=n_gated_clicks,
        n_total_rhombics=n_total_rhombics,
        n_gated_rhombics=n_gated_rhombics,
    )


def per_model_metric(errs: list[float]) -> dict:
    e = np.asarray(errs, dtype=float)
    if e.size == 0:
        return {"n": 0}
    out = {
        "n": int(e.size),
        "mean": float(e.mean()),
        "median": float(np.median(e)),
        "p90": float(np.percentile(e, 90)),
        "p95": float(np.percentile(e, 95)),
    }
    for thr in PCK_THR:
        out[f"pck@{thr}px"] = float((e <= thr).mean())
    return out


def category(name: str) -> str:
    return "A" if name.startswith("A") else "B" if name.startswith("B") else "C"


def write_md(report: dict, out_path: str):
    lines = []
    push = lines.append
    push("# Per-Model Evaluation against v2 GT-ID Labeling\n")
    push(f"- GT source: `runs/full/_real_eval/AC_only/assign/*.npz` (consensus from "
         f"6 A+C models, M=6, threshold 15 px)")
    push(f"- Total instances: **{report['n_total_rhombics']}** rhombics, "
         f"**{report['n_total_clicks']}** GT clicks")
    push(f"- Tier-1 confident subset: **{report['n_gated_rhombics']}** rhombics, "
         f"**{report['n_gated_clicks']}** GT clicks "
         f"(inliers ≥ 5/6 AND assign_mean ≤ 15 px)\n")
    push("Per-click metric: ‖GT_click − M.template[σ_M[assigned_id]]‖₂ in original-image pixels.\n")
    push("Lower is better for mean/median/p90/p95; higher is better for PCK.\n")

    # Two tables: full and gated
    for tag, key in [("Full set (all rhombics)", "full"),
                     ("Tier-1 confident subset", "gated")]:
        errs_dict = report[f"{key}_errs"]
        push(f"## {tag}\n")
        hdr = ["model", "cat", "n", "mean(px)", "med(px)", "p90", "p95"] + [f"PCK@{t}" for t in PCK_THR]
        push("| " + " | ".join(hdr) + " |")
        push("|" + "|".join(["---"] * len(hdr)) + "|")
        # rows sorted by category then name
        for mn in sorted(report["model_names"], key=lambda n: (category(n), n)):
            m = per_model_metric(errs_dict[mn])
            if m["n"] == 0:
                continue
            cells = [mn, category(mn), str(m["n"]),
                     f"{m['mean']:.2f}", f"{m['median']:.2f}",
                     f"{m['p90']:.2f}", f"{m['p95']:.2f}"]
            for t in PCK_THR:
                cells.append(f"{m[f'pck@{t}px']*100:.1f}%")
            push("| " + " | ".join(cells) + " |")

        # Category roll-up
        push("")
        push("**Category roll-up (mean of per-model means):**\n")
        cat_means = {"A": [], "B": [], "C": []}
        cat_pck10 = {"A": [], "B": [], "C": []}
        cat_pck15 = {"A": [], "B": [], "C": []}
        for mn in report["model_names"]:
            m = per_model_metric(errs_dict[mn])
            if m["n"] == 0:
                continue
            cat_means[category(mn)].append(m["mean"])
            cat_pck10[category(mn)].append(m["pck@10px"])
            cat_pck15[category(mn)].append(m["pck@15px"])
        push("| cat | mean(px) | PCK@10 | PCK@15 |")
        push("|---|---|---|---|")
        for c in ["A", "B", "C"]:
            if not cat_means[c]:
                continue
            push(f"| {c} | {np.mean(cat_means[c]):.2f} | "
                 f"{np.mean(cat_pck10[c])*100:.1f}% | "
                 f"{np.mean(cat_pck15[c])*100:.1f}% |")
        push("")

    # σ-residual diagnostic
    push("## σ alignment residual per model (diagnostic)\n")
    push("Higher = the model's template more often disagrees in *shape* with the consensus.\n")
    push("| model | cat | mean σ-residual (px) | median |")
    push("|---|---|---|---|")
    for mn in sorted(report["model_names"], key=lambda n: (category(n), n)):
        sr = np.array(report["full_sigma_resid"][mn])
        push(f"| {mn} | {category(mn)} | {sr.mean():.2f} | {np.median(sr):.2f} |")
    push("")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier-min-inliers", type=int, default=5,
                    help="min voting inliers (out of 6) for Tier-1")
    ap.add_argument("--tier-max-assign-mean", type=float, default=15.0,
                    help="max assign_mean_cost px for Tier-1")
    args = ap.parse_args()

    rep = evaluate(args.tier_min_inliers, args.tier_max_assign_mean)

    # Compute summarised metrics for JSON
    summary = {
        "config": {
            "tier_min_inliers": args.tier_min_inliers,
            "tier_max_assign_mean_px": args.tier_max_assign_mean,
        },
        "n_total_rhombics": rep["n_total_rhombics"],
        "n_total_clicks": rep["n_total_clicks"],
        "n_gated_rhombics": rep["n_gated_rhombics"],
        "n_gated_clicks": rep["n_gated_clicks"],
        "per_model_full": {n: per_model_metric(rep["full_errs"][n])
                           for n in rep["model_names"]},
        "per_model_gated": {n: per_model_metric(rep["gated_errs"][n])
                            for n in rep["model_names"]},
        "sigma_residual_mean_px": {n: float(np.mean(rep["full_sigma_resid"][n]))
                                   for n in rep["model_names"]},
        "sigma_residual_median_px": {n: float(np.median(rep["full_sigma_resid"][n]))
                                     for n in rep["model_names"]},
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"wrote {OUT_JSON}")
    write_md(rep, OUT_MD)


if __name__ == "__main__":
    main()
