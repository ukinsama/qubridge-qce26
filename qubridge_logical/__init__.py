"""Vendored subset of QuBridge needed by the logical-teleportation walkthroughs.

Self-contained copy of the modules used in `notebooks/walkthrough_logical_*.ipynb`
so the artifact does not depend on the upstream `Qubridge/` source tree.

Origin (upstream paths):
  - circuits.py            ← services/mode3_computation.py + utils/circuit.py
  - static_backend.py      ← services/static_backend.py
  - qubit_selection.py     ← experiments/common/qubit_selection.py
  - noise_utils.py         ← experiments/common/noise_utils.py
  - lindblad_synthesis.py  ← experiments/common/lindblad_synthesis.py

Calibration data is read from `qubridge-artifact/calibration/`, not the upstream
`static/data/backend_errors.json`.
"""
from .circuits import (
    create_deferred_logical_teleportation_circuit,
    create_logical_teleportation_circuit,
    create_physical_teleportation_circuit,
    create_deferred_teleportation_circuit,
    compute_dm_fidelity_logical,
    compute_dm_fidelity_physical,
)
from .static_backend import load_static_backend_data
from .qubit_selection import select_qubits_qubridge_logical
from .noise_utils import create_reduced_noise_model, DEFAULT_GATE_PULSE_MAP

__all__ = [
    "create_deferred_logical_teleportation_circuit",
    "create_logical_teleportation_circuit",
    "create_physical_teleportation_circuit",
    "create_deferred_teleportation_circuit",
    "compute_dm_fidelity_logical",
    "compute_dm_fidelity_physical",
    "load_static_backend_data",
    "select_qubits_qubridge_logical",
    "create_reduced_noise_model",
    "DEFAULT_GATE_PULSE_MAP",
]
