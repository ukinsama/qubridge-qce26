"""Regenerate the four data-driven figures from the bundled CSVs.

Outputs PNG copies of:
    figures/fig2b_pipeline_convergence.pdf  (Pipeline ablation)
    figures/fig3b_filter_band.pdf           (L2 filter band)
    figures/fig4b_pergate_comparison.pdf    (L3 per-gate pulse comparison)
    figures/fig5_noise_scaling.pdf          (Encoded vs physical, noise sweep)

Reference PDFs are shipped in `figures/`. This script regenerates the
data-bearing parts so reviewers can confirm the curves come from the CSVs.

Usage:
    python scripts/plot_figures.py [--out OUT_DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"


def plot_fig3b_filter_band(out_dir: Path) -> Path:
    df = pd.read_csv(DATA / "table2_l2_filter.csv")
    order = df["filter_label"].drop_duplicates().tolist()
    grouped = df.groupby("filter_label").agg(
        worst=("fidelity", "min"),
        best=("fidelity", "max"),
    ).reindex(order)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = range(len(grouped))
    ax.fill_between(x, grouped["worst"], grouped["best"],
                    color="#94A3B8", alpha=0.3, label="Fidelity band")
    ax.plot(x, grouped["best"], "o-", color="#16A34A", label="Best F")
    ax.plot(x, grouped["worst"], "s-", color="#2563EB", label="Worst F")
    ax.set_xticks(x)
    ax.set_xticklabels(order, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Fidelity F")
    ax.set_title("Fig 3(b): L2 filter cascade")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = out_dir / "fig3b_filter_band.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_fig5_noise_scaling(out_dir: Path) -> Path:
    df = pd.read_csv(DATA / "table3_encoded_teleportation.csv")
    grouped = df.groupby(["state_label", "noise_scale", "circuit_type"]).agg(
        F=("fidelity", "mean"),
    ).reset_index()

    states = sorted(grouped["state_label"].unique())
    fig, axes = plt.subplots(len(states), 1, figsize=(6, 3.0 * len(states)),
                             sharex=True)
    if len(states) == 1:
        axes = [axes]

    for ax, state in zip(axes, states):
        sub = grouped[grouped["state_label"] == state]
        for ctype, color in [("physical", "#DC2626"), ("logical", "#16A34A")]:
            cs = sub[sub["circuit_type"] == ctype]
            ax.plot(cs["noise_scale"], cs["F"], "o-", color=color,
                    label=f"{ctype.capitalize()}")
        ax.set_title(f"State {state}")
        ax.set_ylabel("Fidelity F")
        ax.grid(axis="y", linestyle=":", alpha=0.5)
        ax.legend(fontsize=8)

    axes[-1].set_xlabel("Noise scale")
    fig.suptitle("Fig 5: Teleportation fidelity under noise scaling")
    fig.tight_layout()
    out = out_dir / "fig5_noise_scaling.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_fig2b_pipeline_convergence(out_dir: Path) -> Path:
    df = pd.read_csv(DATA / "table1_pipeline_trace.csv")
    grouped = df.groupby(["stage", "stage_label"]).agg(
        worst=("fidelity", "min"),
        best=("fidelity", "max"),
    ).reset_index().sort_values("stage")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = range(len(grouped))
    ax.fill_between(x, grouped["worst"], grouped["best"],
                    color="#94A3B8", alpha=0.3, label="Fidelity band")
    ax.plot(x, grouped["best"], "o-", color="#16A34A", label="Best F")
    ax.plot(x, grouped["worst"], "s-", color="#2563EB", label="Worst F")
    ax.set_xticks(x)
    ax.set_xticklabels(grouped["stage_label"], rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Fidelity F")
    ax.set_title("Fig 2(b): Pipeline-band convergence")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()
    out = out_dir / "fig2b_pipeline_convergence.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_fig4b_pergate(out_dir: Path) -> Path:
    df = pd.read_csv(DATA / "table1_pipeline_trace.csv")
    sub = df[df["stage_label"].str.contains("L3", case=False, na=False)
             | (df["stage"] == df["stage"].max())]
    if sub.empty:
        sub = df.copy()
    grouped = sub.groupby("pulse_config").agg(
        F_mean=("fidelity", "mean"),
        F_min=("fidelity", "min"),
        F_max=("fidelity", "max"),
    ).sort_values("F_mean")

    fig, ax = plt.subplots(figsize=(6, 3.5))
    y = range(len(grouped))
    ax.barh(y, grouped["F_mean"], color="#2563EB", alpha=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels(grouped.index, fontsize=8)
    ax.set_xlabel("Fidelity F")
    ax.set_title("Fig 4(b): L3 per-gate pulse comparison")
    ax.grid(axis="x", linestyle=":", alpha=0.5)
    for i, (lo, mean, hi) in enumerate(zip(grouped["F_min"], grouped["F_mean"], grouped["F_max"])):
        ax.plot([lo, hi], [i, i], color="#0F172A", alpha=0.6)
    fig.tight_layout()
    out = out_dir / "fig4b_pergate_comparison.png"
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default=str(ROOT / "figures"),
                        help="Output directory for PNG copies (default: figures/)")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = [
        plot_fig2b_pipeline_convergence(out_dir),
        plot_fig3b_filter_band(out_dir),
        plot_fig4b_pergate(out_dir),
        plot_fig5_noise_scaling(out_dir),
    ]
    for p in paths:
        print(f"saved: {p}")


if __name__ == "__main__":
    main()
