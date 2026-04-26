"""
軽量ノイズモデル生成ユーティリティ（Lindblad synthesis ベース）

選択した physical qubit のみに、較正済み RB 誤差を anchor として Pauli-Lindblad
生成子に物理的に配分したノイズを注入する。実装は ``lindblad_synthesis`` モジュールに
委譲し、ここでは後方互換 API ``create_reduced_noise_model`` / ``pulse_leakage_error``
などを提供する。

理論的背景:
  - Malekakhlagh et al., npj QI 2025 (arXiv:2502.03462) — Magnus M1 Lindblad synthesis
  - van den Berg et al., Nat. Phys. 2023 — sparse Pauli-Lindblad model
  - Gambetta et al., PRA 83, 012308 (2011) — 1Q パルス形リーケージ摂動

各 gate は単一の QuantumError = thermal_relaxation_error (厳密 Kraus) ∘
PauliLindbladError (物理配分 RB 残差 + 静的 ZZ) として構築する。Qiskit Aer の
NoiseModel に attach できる形式。
"""
import numpy as np
from typing import List

from .lindblad_synthesis import (
    build_lindblad_noise_model as _build_lindblad_noise_model,
)


# =============================================================================
# Physical constants & pulse shape parameters
# =============================================================================

# --- 1Q leakage: Gambetta et al. (PRA 83, 012308, 2011) ---
# L = C2*(Omega/alpha)^2 + C4*(Omega/alpha)^4
# DRAG cancels 2nd-order (C2=0); only 4th-order remains.
PULSE_LEAKAGE_C2 = {
    "drag": 0.0,              # Q-component cancels O(Omega/alpha)^2
    "gaussian_square": 1.23,  # ~pi^2/8
    "gaussian": 1.23,
    "square": 2.47,           # pi^2/4
}
PULSE_LEAKAGE_C4 = {
    "drag": 0.05,             # residual O(Omega/alpha)^4
    "gaussian_square": 1.5,
    "gaussian": 1.5,
    "square": 6.0,
}

# --- 2Q gate: pulse-shape-dependent control error ---
# Cross-resonance requires flat-top for precise phase accumulation.
# GaussianSquare = smooth ramp → minimal spectral leakage in 2Q subspace.
# Square = sharp edges → spectral leakage → ZX/ZY crosstalk.
# Values: multiplicative scaling of the calibrated 2Q error rate.
PULSE_2Q_ERROR_SCALE = {
    "gaussian_square": 1.0,   # optimal for cross-resonance (IBM default)
    "gaussian": 1.0,
    "drag": 1.4,              # 1Q-optimized, suboptimal for 2Q
    "square": 1.8,            # sharp edges → spectral leakage
}

# --- Readout: pulse shape effect on measurement SNR ---
# Square = flat-top maximizes integration time → best SNR.
# Gaussian = ramp loses integration time → worse discrimination.
# DRAG = Q-component adds noise, no readout benefit.
# Values: multiplicative scaling of calibrated readout error.
PULSE_READOUT_ERROR_SCALE = {
    "square": 1.0,            # optimal for readout (IBM default)
    "gaussian_square": 1.2,   # slight SNR loss from ramps
    "gaussian": 1.5,          # significant integration time loss
    "drag": 1.8,              # Q-component adds noise
}

# --- Gate duration scaling per pulse shape ---
# Calibrated duration relative to backend's gate_time:
#   DRAG: 1.0x — IBM standard (32ns SX). Compact due to leakage suppression.
#   GaussianSquare: 1.3x — ramp-up/ramp-down adds time.
#   Square: 0.8x — rectangular pulse is shorter, but suffers leakage.
PULSE_CALIBRATED_DURATION = {
    "drag": 1.0,
    "gaussian_square": 1.3,
    "gaussian": 1.3,
    "square": 0.8,
}

DEFAULT_ANHARMONICITY_GHZ = 0.330  # |alpha| for IBM Heron/Eagle transmons
DEFAULT_COUPLING_GHZ = 0.002       # ~2 MHz nearest-neighbor exchange coupling

# Default per-gate pulse shape assignment (IBM backend calibration)
DEFAULT_GATE_PULSE_MAP = {
    "sx": "drag",              # DRAG for 1Q gates (leakage suppression)
    "x": "drag",
    "cz": "gaussian_square",   # GaussianSquare for 2Q cross-resonance
    "ecr": "gaussian_square",
    "measure": "square",       # Square for readout (flat integration window)
    "id": "drag",
    "rz": "drag",              # virtual-Z, no actual pulse (duration=0)
}

# =============================================================================
# Per-gate error models (pure functions — kept for legacy consumers like
# l3_tuning_figure.py analytical fidelity estimates)
# =============================================================================

def pulse_leakage_error(pulse_shape: str, gate_time_s: float,
                         anharmonicity_ghz: float = DEFAULT_ANHARMONICITY_GHZ) -> float:
    """1Q leakage error from perturbation theory (Gambetta et al. 2011).

    L = C2*(Omega/alpha)^2 + C4*(Omega/alpha)^4
    where Omega = (pi/2) / t_gate.
    DRAG: C2=0 (Q-component cancels 2nd-order), only 4th-order remains.
    """
    gate_time_ns = gate_time_s * 1e9
    omega = (np.pi / 2) / gate_time_ns          # rad/ns
    alpha_ns = 2 * np.pi * anharmonicity_ghz     # rad/ns
    ratio = omega / alpha_ns

    C2 = PULSE_LEAKAGE_C2.get(pulse_shape, 0.0)
    C4 = PULSE_LEAKAGE_C4.get(pulse_shape, 0.05)
    return min(C2 * ratio**2 + C4 * ratio**4, 0.75)


def decoherence_error(gate_time_s: float, t1: float, t2: float) -> float:
    """Gate error from decoherence (exact Lindblad coefficients, arXiv:2302.13885).

    epsilon = (2/3)*gamma_1*t + (4/3)*gamma_phi*t
    where gamma_1 = 1/T1, gamma_phi = 1/T2 - 1/(2*T1).
    """
    if t1 <= 0 or t2 <= 0:
        return 0.0
    gamma_1 = 1.0 / t1
    gamma_phi = max(0.0, 1.0 / t2 - 1.0 / (2 * t1))
    return min((2.0 / 3) * gamma_1 * gate_time_s + (4.0 / 3) * gamma_phi * gate_time_s, 0.75)


def zz_coupling_error(gate_time_s: float, detuning_ghz: float,
                       anharmonicity_ghz: float = DEFAULT_ANHARMONICITY_GHZ,
                       coupling_ghz: float = DEFAULT_COUPLING_GHZ) -> float:
    """ZZ phase error from static coupling (Sheldon et al. PRA 93, 2016).

    omega_ZZ = J^2 * alpha / (Delta^2 - alpha^2)
    epsilon_ZZ = |omega_ZZ| * t_gate
    """
    if detuning_ghz < 0.01:
        return 0.01  # near-degenerate — model breaks down
    denom = detuning_ghz**2 - anharmonicity_ghz**2
    if abs(denom) < 1e-6:
        return 0.01  # spectral collision regime
    omega_zz = coupling_ghz**2 * anharmonicity_ghz / denom
    return min(abs(omega_zz) * 2 * np.pi * 1e9 * gate_time_s, 0.75)


# =============================================================================
# Reduced noise model — delegates to Lindblad synthesis
# =============================================================================

def create_reduced_noise_model(backend_name: str, physical_qubits: List[int],
                                gate_pulse_map: dict | None = None,
                                duration_scale: float = 1.0,
                                noise_scale: float = 1.0):
    """Lindblad synthesis ベースの軽量ノイズモデル（後方互換 API）。

    実装は ``lindblad_synthesis.build_lindblad_noise_model`` に委譲する。
    各 gate は ``thermal_relaxation_error`` ∘ ``PauliLindbladError`` の
    composite QuantumError として構築され、RB 較正値を anchor に
    物理的に配分された Pauli 生成子を持つ。

    Args:
        backend_name: 静的バックエンド名（例: "ibm_torino"）
        physical_qubits: 対象とする物理量子ビットのリスト（virtual 0..N-1 にマップ）
        gate_pulse_map: gate タイプ別のパルス形指定（DEFAULT_GATE_PULSE_MAP を上書き）
        duration_scale: ゲート duration 乗数（duration engineering 実験用）
        noise_scale: 全較正レート × noise_scale、T1,T2 ÷ noise_scale。0.0 で empty。
    """
    return _build_lindblad_noise_model(
        backend_name=backend_name,
        physical_qubits=physical_qubits,
        gate_pulse_map=gate_pulse_map,
        duration_scale=duration_scale,
        noise_scale=noise_scale,
    )


