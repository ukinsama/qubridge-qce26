"""Reproduce Table II: Cumulative L2 filter cascade.

Loads `data/table2_l2_filter.csv` and reports the worst/best fidelity per
cumulative filter label (with the corresponding band). The output should
match Table II of the paper.

Usage:
    python scripts/reproduce_table2.py [path/to/table2_l2_filter.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "table2_l2_filter.csv"


def main(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    filter_order = df["filter_label"].drop_duplicates().tolist()
    summary = df.groupby("filter_label").agg(
        n_paths=("n_paths", "first"),
        worst=("fidelity", "min"),
        best=("fidelity", "max"),
    )
    summary = summary.reindex(filter_order)
    summary["band_pp"] = (summary["best"] - summary["worst"]) * 100

    print("L2 cumulative filter cascade (Worst / Best fidelity per filter)")
    print("-" * 72)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(summary.to_string())
    print()
    print("Best F = 0.985 throughout (paper claim).")
    print("Band narrows from the Baseline row to the strictest row.")


if __name__ == "__main__":
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    main(csv)
