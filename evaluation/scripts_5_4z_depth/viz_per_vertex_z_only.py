"""3×3 grid: per-vertex |Z error| (red bars) with mean |X|, |Y| reference lines.

Paper-oriented redesign of `z_per_vertex_xyz_error.png`.
Two messages this figure delivers:
    (1) Z dominates — Z bars sit far above the X/Y reference lines.
    (2) Z error is FLAT across all 14 vertices — supports the
        "single per-instance offset suffices" framing.

Output:
    results/5.4Z_depth/visualizations/z_analysis/fig_per_vertex_z_only.{png,pdf}
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("<PROJECT_ROOT>/evaluation")
METRICS = ROOT / "results" / "5.4Z_depth" / "metrics" / "v3_9entry_metrics.json"
OUT = ROOT / "results" / "5.4Z_depth" / "visualizations" / "z_analysis"
OUT.mkdir(parents=True, exist_ok=True)

ENTRIES = [
    ("A1_gdrnet",                  "A1 GDR-Net"),
    ("A2_sc6d",                    "A2 SC6D"),
    ("A3_hccepose",                "A3 HccePose"),
    ("B1_rede",                    "B1 REDE (raw)"),
    ("B2_uni6d",                   "B2 Uni6D (raw)"),
    ("B3_ffb6d",                   "B3 FFB6D (raw)"),
    ("C1_megapose_official",       "C1 MegaPose"),
    ("C2_gigapose_official",       "C2 GigaPose"),
    ("C3_foundationpose_official", "C3 FoundPose"),
]

Z_COLOR = "#c0392b"
X_COLOR = "#2980b9"
Y_COLOR = "#e67e22"


def main() -> None:
    M = json.load(open(METRICS))
    n = M["n_instances"]

    fig, axes = plt.subplots(3, 3, figsize=(13, 9), sharex=True, sharey=True)

    # global y-limit for shared scale
    all_z = []
    for stem, _ in ENTRIES:
        arr = np.array(M["entries"][stem]["per_vertex_xyz"])
        all_z.append(arr[:, 2])
    ymax = float(max(z.max() for z in all_z)) * 1.12

    for ax, (stem, label) in zip(axes.flat, ENTRIES):
        per_v = np.array(M["entries"][stem]["per_vertex_xyz"])  # (14, 3)
        z = per_v[:, 2]
        x_mean = float(per_v[:, 0].mean())
        y_mean = float(per_v[:, 1].mean())
        z_mean = float(z.mean())

        # Z bars
        v_idx = np.arange(14)
        ax.bar(v_idx, z, width=0.75, color=Z_COLOR, alpha=0.88,
               edgecolor="#7e1f12", linewidth=0.6, zorder=3, label="|Z| per vertex")

        # X / Y reference lines (mean over 14 vertices)
        ax.axhline(x_mean, color=X_COLOR, lw=1.4, ls="--", alpha=0.95, zorder=4,
                   label=f"mean |X| = {x_mean:.1f}")
        ax.axhline(y_mean, color=Y_COLOR, lw=1.4, ls="--", alpha=0.95, zorder=4,
                   label=f"mean |Y| = {y_mean:.1f}")

        # Z mean horizontal line (subtle)
        ax.axhline(z_mean, color="#7e1f12", lw=0.9, ls=":", alpha=0.55, zorder=2)

        # Title + annotation box: ratio
        ratio = z_mean / max(0.5 * (x_mean + y_mean), 1e-9)
        ax.set_title(label, fontsize=11.5, fontweight="bold")
        ax.text(0.97, 0.97,
                f"mean |Z| = {z_mean:.1f}\n"
                f"Z / mean(X,Y) = {ratio:.2f}×",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=8.7, family="monospace",
                bbox=dict(facecolor="white", edgecolor="#aaa",
                          alpha=0.94, pad=3, linewidth=0.7))

        # legend on first panel only
        if (stem, label) == ENTRIES[0]:
            ax.legend(loc="upper left", fontsize=8.0, framealpha=0.92,
                      handlelength=2.0, handletextpad=0.4, borderpad=0.4)

        ax.set_ylim(0, ymax)
        ax.set_xticks(v_idx)
        ax.tick_params(labelsize=9, colors="#444")
        ax.grid(axis="y", alpha=0.25)
        ax.grid(axis="x", visible=False)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("bottom", "left"):
            ax.spines[s].set_color("#888"); ax.spines[s].set_linewidth(0.7)

    for ax in axes[-1]:
        ax.set_xlabel("Vertex index (0–13)", fontsize=10)
    for ax in axes[:, 0]:
        ax.set_ylabel("|error| (px)", fontsize=10)

    fig.suptitle(
        "Per-vertex |Z error| across 9 baselines  "
        f"(n = {n:,}; X/Y shown as mean reference lines)\n"
        "Z bars are flat across vertices ⇒ a single per-instance Z offset captures the dominant error.",
        fontsize=12, y=0.998, fontweight="bold")
    fig.tight_layout()

    png = OUT / "fig_per_vertex_z_only.png"
    pdf = OUT / "fig_per_vertex_z_only.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight")
    fig.savefig(pdf,            bbox_inches="tight")
    plt.close(fig)
    print(f"saved: {png.relative_to(ROOT)}")
    print(f"saved: {pdf.relative_to(ROOT)}")

    # also print per-model summary
    print("\n--- per-model |X|, |Y|, |Z| means + ratio ---")
    print(f"  {'Model':<19} {'|X|':>7} {'|Y|':>7} {'|Z|':>7}  Z/(X+Y)/2")
    for stem, label in ENTRIES:
        per_v = np.array(M["entries"][stem]["per_vertex_xyz"])
        x_m, y_m, z_m = per_v.mean(axis=0)
        ratio = z_m / (0.5 * (x_m + y_m))
        print(f"  {label:<19} {x_m:>7.2f} {y_m:>7.2f} {z_m:>7.2f}      {ratio:.2f}×")


if __name__ == "__main__":
    main()
