"""Minimal load + plot demo for the L2 filter cascade CSV (Table II / Fig 3b).

Usage:
    python scripts/plot_example.py data/table2_l2_filter.csv

Loads the CSV, computes Worst/Best F per filter stage across both input
states (|+>, |1>), and plots the band-narrowing trajectory analogous to
Fig 3(b) in the paper.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    filter_order = df["filter_label"].drop_duplicates().tolist()
    df["filter_idx"] = df["filter_label"].map({lbl: i for i, lbl in enumerate(filter_order)})

    grouped = df.groupby("filter_label").agg(
        worst=("fidelity", "min"),
        best=("fidelity", "max"),
    )
    grouped = grouped.reindex(filter_order)
    grouped["band_pp"] = (grouped["best"] - grouped["worst"]) * 100

    print(grouped)

    fig, ax = plt.subplots(figsize=(6, 3.5))
    x = range(len(grouped))
    ax.fill_between(x, grouped["worst"], grouped["best"],
                    color="#94A3B8", alpha=0.3, label="Fidelity band")
    ax.plot(x, grouped["best"], "o-", color="#16A34A", label="Best F (L3=All Sq.)")
    ax.plot(x, grouped["worst"], "s-", color="#2563EB", label="Worst F")
    ax.set_xticks(x)
    ax.set_xticklabels(filter_order, rotation=20, ha="right", fontsize=8)
    ax.set_ylabel("Fidelity F")
    ax.set_title("L2 Cumulative Filtering — Fidelity Band")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", linestyle=":", alpha=0.5)
    fig.tight_layout()

    out = Path(csv_path).with_suffix(".png")
    fig.savefig(out, dpi=150)
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print(__doc__)
        sys.exit(1)
    main(sys.argv[1])
