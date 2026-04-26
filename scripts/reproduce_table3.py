"""Reproduce Table III: Encoded teleportation across noise scales.

Loads `data/table3_encoded_teleportation.csv` and pivots on
(state_label, noise_scale) to report Physical F, Logical F, and the
syndrome acceptance fraction. The output should match Table III of the
paper.

Usage:
    python scripts/reproduce_table3.py [path/to/table3_encoded_teleportation.csv]
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

DEFAULT_CSV = (
    Path(__file__).resolve().parents[1] / "data" / "table3_encoded_teleportation.csv"
)


def main(csv_path: Path) -> None:
    df = pd.read_csv(csv_path)

    grouped = df.groupby(
        ["state_label", "noise_scale", "circuit_type"], sort=True
    ).agg(
        F=("fidelity", "mean"),
        accept=("error_detection_rate", "mean"),
    )

    pivoted = grouped.unstack("circuit_type")
    pivoted.columns = ["_".join(filter(None, col)).strip() for col in pivoted.columns]
    pivoted = pivoted[
        [c for c in ["F_physical", "F_logical", "accept_logical"] if c in pivoted.columns]
    ]
    pivoted["delta_F"] = pivoted["F_logical"] - pivoted["F_physical"]
    pivoted["throughput_logical"] = pivoted["F_logical"] * pivoted["accept_logical"]

    print("Physical / Logical fidelity and syndrome acceptance vs noise scale")
    print("-" * 72)
    pd.set_option("display.float_format", lambda v: f"{v:.4f}")
    print(pivoted.to_string())
    print()
    print("Sign of delta_F is state-dependent: |+> stays negative,")
    print("|1> stays positive (logical > physical conditional on syndrome).")


if __name__ == "__main__":
    csv = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_CSV
    main(csv)
