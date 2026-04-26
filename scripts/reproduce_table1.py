"""Reproduce Table I: Pipeline-band narrowing (per stage).

Loads `data/table1_pipeline_trace.csv` and reports Min / Avg / Max fidelity
plus the band width (Max - Min) at each pipeline stage. The output should
match the values reported in Table I of the paper.

Usage:
    python scripts/reproduce_table1.py [path/to/table1_pipeline_trace.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_CSV = Path(__file__).resolve().parents[1] / "data" / "table1_pipeline_trace.csv"


def main(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    summary = df.groupby(["stage", "stage_label"], sort=True).agg(
        n=("fidelity", "size"),
        min=("fidelity", "min"),
        avg=("fidelity", "mean"),
        max=("fidelity", "max"),
    )
    summary["band_pp"] = (summary["max"] - summary["min"]) * 100

    print("Pipeline-band narrowing across stages")
    print("-" * 72)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(summary.to_string())
    print()
    print("Each row = one pipeline configuration cluster. `band_pp` is the")
    print("width of the fidelity distribution in percentage points.")


if __name__ == "__main__":
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    main(csv)
