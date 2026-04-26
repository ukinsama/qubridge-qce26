"""Calibration loader for the vendored walkthrough modules.

Adapted from `services/static_backend.py` (upstream QuBridge). Trimmed to
``load_static_backend_data`` and rebound to the artifact's
``calibration/ibm_torino_*.json`` snapshot, which uses a single-backend
schema ({"backend": {...}}) rather than the upstream multi-backend
({"backends": {name: {...}}}) format.
"""
from __future__ import annotations

import json
import logging
import os
import threading
from typing import Any, Dict, Optional, cast

logger = logging.getLogger(__name__)

# --- Locate the artifact's calibration directory --------------------------------
# qubridge_logical/ lives directly under qubridge-artifact/, so the calibration
# folder is one directory up.
_PKG_DIR = os.path.dirname(os.path.abspath(__file__))
_ARTIFACT_ROOT = os.path.dirname(_PKG_DIR)
_CALIBRATION_DIR = os.path.join(_ARTIFACT_ROOT, "calibration")

_static_data_cache: Dict[str, Dict[str, Any]] = {}
_cache_lock = threading.Lock()


def _find_calibration_file(backend_name: str) -> str:
    """Locate the calibration JSON for `backend_name` inside calibration/.

    Accepts either ``<backend>.json`` or ``<backend>_<date>.json``.
    """
    if not os.path.isdir(_CALIBRATION_DIR):
        raise FileNotFoundError(
            f"Calibration directory not found: {_CALIBRATION_DIR}"
        )
    candidates = []
    for fname in sorted(os.listdir(_CALIBRATION_DIR)):
        if not fname.endswith(".json"):
            continue
        stem = fname[: -len(".json")]
        if stem == backend_name or stem.startswith(backend_name + "_"):
            candidates.append(os.path.join(_CALIBRATION_DIR, fname))
    if not candidates:
        available = sorted(
            f for f in os.listdir(_CALIBRATION_DIR) if f.endswith(".json")
        )
        raise ValueError(
            f"No calibration file for backend '{backend_name}' under "
            f"{_CALIBRATION_DIR}. Available: {available}"
        )
    # Prefer the lexicographically last (= most recent date suffix)
    return candidates[-1]


def load_static_backend_data(backend_name: str) -> Dict[str, Any]:
    """Load the cached IBM backend snapshot for `backend_name`.

    Returns the ``backend`` payload with the standard QuBridge schema
    (``errors``, ``coupling_map``, ``gate_properties``, ``qubit_properties``,
    ``basis_gates``, ``n_qubits``, ``timing``, ``instruction_durations``).
    """
    with _cache_lock:
        cached = _static_data_cache.get(backend_name)
        if cached is not None:
            return cached

        path = _find_calibration_file(backend_name)
        logger.debug("Loading calibration snapshot from %s", path)
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)

        # Artifact format: {"backend": {...}} (single backend per file).
        # Upstream format: {"backends": {name: {...}}}. Accept both.
        if "backend" in raw and isinstance(raw["backend"], dict):
            backend_data = raw["backend"]
        elif "backends" in raw and isinstance(raw["backends"], dict):
            backends = raw["backends"]
            if backend_name not in backends:
                raise ValueError(
                    f"Backend '{backend_name}' not in {path}. "
                    f"Available: {sorted(backends.keys())}"
                )
            backend_data = backends[backend_name]
        else:
            raise ValueError(
                f"Unexpected calibration JSON schema in {path}; expected "
                f"top-level 'backend' or 'backends' key."
            )

        _static_data_cache[backend_name] = cast(Dict[str, Any], backend_data)
        return _static_data_cache[backend_name]


def clear_cache() -> None:
    with _cache_lock:
        _static_data_cache.clear()
