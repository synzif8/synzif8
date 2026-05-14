"""Paper figure C: per-instance Z offset distribution histograms (3x3 grid).

For each of 9 baseline models on the full SYN test set (n=11,379), plot the
distribution of per-instance Z offsets:
    dz_instance = mean(pred_verts[:, 2] - gt_verts[:, 2])

Output: figures/fig_C_offset_distribution.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("<PROJECT_ROOT>/evaluation")
PER_MODEL = ROOT / "results" / "per_model"
OUT = ROOT / "figures"

# (dump-name, label, category, pred-field) — Layout in row-major order
ENTRIES = [
    ("A1_gdrnet",                  "A1 GDR-Net",  "A", "pred_verts"),
    ("A2_sc6d",                    "A2 SC6D",     "A", "pred_verts"),
    ("A3_hccepose",                "A3 HccePose", "A", "pred_verts"),
    ("B1_rede",                    "B1 REDE",     "B", "kp3d_raw"),
    ("B2_uni6d",                   "B2 Uni6D",    "B", "kp3d_raw"),
    ("B3_ffb6d",                   "B3 FFB6D",    "B", "kp3d_raw"),
    ("C1_megapose_official",       "C1 MegaPose", "C", "pred_verts"),
    ("C2_gigapose_official",       "C2 GigaPose", "C", "pred_verts"),
    ("C3_foundationpose_official", "C3 FoundPose","C", "pred_verts"),
]
CAT_COLOR = {"A": "#4f9ed3", "B": "#e67e22", "C": "#16a085"}


def _clean(ax) -> None:
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#888"); ax.spines[spine].set_linewidth(0.7)
    ax.tick_params(labelsize=9, colors="#444")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    rows = []
    for dump, label, cat, field in ENTRIES:
        rec = json.load(open(PER_MODEL / f"{dump}_dump.json"))["records"]
        dz = []
        for r in rec:
            gt = np.asarray(r["gt_verts"], dtype=np.float64)
            pr = np.asarray(r[field], dtype=np.float64)
            if gt.shape != (14, 3) or pr.shape != (14, 3):
                continue
            dz.append(float((pr[:, 2] - gt[:, 2]).mean()))
        dz = np.asarray(dz)
        rows.append({"label": label, "cat": cat, "dz": dz,
                     "mean": float(dz.mean()), "std": float(dz.std())})
        print(f"  {label:<14}  n={len(dz)}  mean={dz.mean():+7.2f}  std={dz.std():6.2f}")

    # Shared x-range: global min/max with 5% padding
    all_dz = np.concatenate([r["dz"] for r in rows])
    xmin, xmax = float(all_dz.min()), float(all_dz.max())
    pad = (xmax - xmin) * 0.05
    xlim = (xmin - pad, xmax + pad)
    bins = np.linspace(xlim[0], xlim[1], 61)  # 60 bins

    # Pre-compute counts to set shared y-limit
    max_count = 0
    counts_per_model = []
    for r in rows:
        c, _ = np.histogram(r["dz"], bins=bins)
        counts_per_model.append(c)
        max_count = max(max_count, int(c.max()))
    ylim = (0, int(max_count * 1.10))

    fig, axes = plt.subplots(3, 3, figsize=(12, 9), sharex=True, sharey=True)

    for ax, r, c in zip(axes.flat, rows, counts_per_model):
        ax.hist(r["dz"], bins=bins, color=CAT_COLOR[r["cat"]],
                alpha=0.70, edgecolor="black", linewidth=0.3)
        ax.axvline(0, color="gray", lw=1.0, ls="--", alpha=0.5)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim)
        ax.set_title(r["label"], fontsize=11, fontweight="bold")
        ax.text(0.97, 0.97,
                f"mean = {r['mean']:+5.1f} px\n"
                f"std  = {r['std']:5.1f} px",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=9, family="monospace",
                bbox=dict(facecolor="white", edgecolor="#aaa",
                          alpha=0.92, pad=3, linewidth=0.7))
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
        _clean(ax)

    for ax in axes[-1]:
        ax.set_xlabel("Per-instance Δz (px)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("Instance count", fontsize=10)

    fig.suptitle(
        f"Distribution of per-instance Z offsets across 9 baseline models  "
        f"(n = {len(rows[0]['dz']):,} instances each)",
        fontsize=13, y=0.995,
    )
    fig.tight_layout()

    png = OUT / "fig_C_offset_distribution.png"
    pdf = OUT / "fig_C_offset_distribution.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf,           bbox_inches="tight")
    plt.close(fig)
    print(f"\nsaved: {png.relative_to(ROOT)}")
    print(f"saved: {pdf.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
