"""Reproduce Table II: Cumulative L2 filter cascade.

Loads `data/table2_l2_filter.csv` and reports the worst fidelity and the
operational band per cumulative filter label. The output should match
Table II of the paper.

Operational definition of Band (per paper Section IV):
    Band = F_PIPELINE_OPT - worst
where F_PIPELINE_OPT = 0.9849 is the best observed pipeline configuration
(L2 = noise-aware path + L3 = per-gate optimal pulse-shape), used as a
practitioner-facing fixed reference. This is *not* the within-row
max - min, because the L2 sweep CSV holds L3 fixed at All Square; using
the within-row max would conflate L2 and L3 contributions.

Usage:
    python scripts/reproduce_table2.py [path/to/table2_l2_filter.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "table2_l2_filter.csv"

# Pipeline optimum (L2 + L3 fully optimal), reported in Table I (+L2+L3 row)
# and used as the fixed reference for Band throughout Section IV.
F_PIPELINE_OPT = 0.9849


def main(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    filter_order = df["filter_label"].drop_duplicates().tolist()
    summary = df.groupby("filter_label").agg(
        n_paths=("n_paths", "first"),
        worst=("fidelity", "min"),
        within_sweep_best=("fidelity", "max"),
    )
    summary = summary.reindex(filter_order)
    summary["band_pp"] = (F_PIPELINE_OPT - summary["worst"]) * 100

    print("L2 cumulative filter cascade")
    print(f"Reference: F_PIPELINE_OPT = {F_PIPELINE_OPT} "
          "(L2 + L3 optimal, fixed reference)")
    print("-" * 72)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(summary.to_string())
    print()
    print("`band_pp` = 100 * (F_PIPELINE_OPT - worst). This matches Table II.")
    print("`within_sweep_best` is shown for context; it is the L2 ceiling")
    print("under L3 = All Square (filled band upper edge in Fig 3b).")


if __name__ == "__main__":
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    main(csv)
