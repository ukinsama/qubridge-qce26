"""
Efficient Lindblad synthesis for noise model construction.

Implements the Magnus M1 (leading-order) Pauli-Lindblad rate synthesis of
Malekakhlagh et al., "Efficient Lindblad synthesis for noise model construction"
(npj Quantum Information, 2025; arXiv:2502.03462).

The paper's approach:
  1. Start from a physical Lindbladian L = L_T1 + L_T2 + L_ZZ + L_control acting
     for duration tau_g during a gate with Hamiltonian H_g.
  2. Move to the interaction frame: L_I(t) = exp(+i H_g t) L exp(-i H_g t).
  3. Integrate via Magnus: Ω_1 = ∫₀^{τ_g} L_I(t) dt (leading order).
  4. Decompose exp(Ω_1) in the sparse Pauli-Lindblad basis:
         Λ(ρ) = exp[Σ_k λ_k (P_k ρ P_k† − ρ)]
     where {P_k} is a small set of Pauli strings dictated by topology.

Our specialization:
  - Decoherence (T1/T2): Qiskit's thermal_relaxation_error implements the exact
    Kraus channel, which is the Magnus-M1 result summed to all orders. We use it
    directly rather than re-deriving Pauli-Lindblad rates.
  - Static ZZ + RB-calibrated residual (control, crosstalk, leakage): we distribute
    the RB-measured total error ε_RB across Pauli-Lindblad generators using physics-
    derived fractions (amp damping → {X, Y, Z}, static ZZ → {ZZ}, rest → uniform).
    This anchors the model to calibrated reality while reflecting physical structure.

The output is a Qiskit AerSimulator-compatible NoiseModel whose per-gate
QuantumError is a single composite channel (`thermal ⊗ ... ⊗ thermal`).compose(
`PauliLindbladError(generators, rates)`) — following the canonical Qiskit Aer
pattern of one composite error per gate.

References:
  - Malekakhlagh, Seif, Puzzuoli, Govia, van den Berg, npj QI 2025
  - van den Berg et al., Nat. Phys. 2023 (sparse Pauli-Lindblad model)
  - Gambetta et al., PRA 83, 012308 (2011) — pulse leakage
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .static_backend import load_static_backend_data


# Inlined from upstream core.config.Config to keep the artifact self-contained.
class _DefaultGateTime:
    DEFAULT_SINGLE_GATE_TIME = 60e-9      # 60 ns (IBM Heron/Eagle SX scale)
    DEFAULT_TWO_QUBIT_GATE_TIME = 400e-9  # 400 ns (CZ scale)


Config = _DefaultGateTime


# =============================================================================
# Pulse-shape constants (shared with noise_utils legacy code)
# =============================================================================

# 1Q leakage coefficients (Gambetta 2011): L = C2·(Ω/α)² + C4·(Ω/α)⁴
PULSE_LEAKAGE_C2 = {"drag": 0.0, "gaussian_square": 1.23, "gaussian": 1.23, "square": 2.47}
PULSE_LEAKAGE_C4 = {"drag": 0.05, "gaussian_square": 1.5, "gaussian": 1.5, "square": 6.0}

# 2Q gate pulse-shape multiplier on calibrated error (cross-resonance optimal = gauss_square)
PULSE_2Q_ERROR_SCALE = {"gaussian_square": 1.0, "gaussian": 1.0, "drag": 1.4, "square": 1.8}

# Readout pulse-shape multiplier (square flat-top maximizes integration)
PULSE_READOUT_ERROR_SCALE = {"square": 1.0, "gaussian_square": 1.2, "gaussian": 1.5, "drag": 1.8}

# Gate duration scale per pulse shape (relative to backend default)
PULSE_CALIBRATED_DURATION = {"drag": 1.0, "gaussian_square": 1.3, "gaussian": 1.3, "square": 0.8}

DEFAULT_ANHARMONICITY_GHZ = 0.330   # |α| for IBM Heron/Eagle transmons
DEFAULT_COUPLING_GHZ = 0.002        # J ~ 2 MHz nearest-neighbor exchange

DEFAULT_GATE_PULSE_MAP = {
    "sx": "drag", "x": "drag", "cz": "gaussian_square", "ecr": "gaussian_square",
    "measure": "square", "id": "drag", "rz": "drag",
}

_FREQ_FALLBACK_GHZ = {
    0: 4.964, 1: 4.838, 2: 5.107, 10: 4.876, 11: 5.021, 12: 4.912,
    73: 5.152, 82: 4.891, 81: 5.043,
}


# =============================================================================
# Pulse-shape physics (kept from legacy; used to derive pulse-shape delta)
# =============================================================================

def pulse_leakage_rate(pulse_shape: str, gate_time_s: float,
                       anharmonicity_ghz: float = DEFAULT_ANHARMONICITY_GHZ) -> float:
    """1Q leakage probability per gate (Gambetta 2011, 2nd+4th order perturbation)."""
    gate_time_ns = gate_time_s * 1e9
    omega = (np.pi / 2) / gate_time_ns
    alpha_ns = 2 * np.pi * anharmonicity_ghz
    ratio = omega / alpha_ns
    C2 = PULSE_LEAKAGE_C2.get(pulse_shape, 0.0)
    C4 = PULSE_LEAKAGE_C4.get(pulse_shape, 0.05)
    return min(C2 * ratio**2 + C4 * ratio**4, 0.75)


def zz_phase_error(gate_time_s: float, detuning_ghz: float,
                   anharmonicity_ghz: float = DEFAULT_ANHARMONICITY_GHZ,
                   coupling_ghz: float = DEFAULT_COUPLING_GHZ) -> float:
    """Static ZZ phase error accumulated during gate time (Sheldon 2016)."""
    if detuning_ghz < 0.01:
        return 0.01
    denom = detuning_ghz**2 - anharmonicity_ghz**2
    if abs(denom) < 1e-6:
        return 0.01
    omega_zz = coupling_ghz**2 * anharmonicity_ghz / denom
    return min(abs(omega_zz) * 2 * np.pi * 1e9 * gate_time_s, 0.75)


# =============================================================================
# Lindblad synthesis: Pauli-Lindblad rate calculations
# =============================================================================

def _clamp_t2(t1: float, t2: float) -> float:
    """Enforce T2 ≤ 2T1 (physical) and T2 ≤ T1 (Aer numerical stability)."""
    return min(t2, 2 * t1, t1)


def _get_gate_time(gate_properties: dict, gate_name: str, phys_q: int) -> float:
    """Calibrated gate duration, falling back to Config default."""
    gt = gate_properties.get(gate_name, {}).get(f"({phys_q},)", {}).get("duration")
    return gt if gt is not None else Config.DEFAULT_SINGLE_GATE_TIME


def decoherence_fractions(T1: float, T2: float) -> Tuple[float, float, float]:
    """Physics-derived fractions for {X, Y, Z} Pauli generators from T1/T2 decoherence.

    From Magnus M1 of amplitude damping + pure dephasing Lindbladians over duration τ:
        rate_X = rate_Y = γ_↓ τ / 4       (amp damping, symmetric X/Y redistribution)
        rate_Z = γ_↓ τ / 4 + γ_φ τ / 2    (amp damping Z-contribution + pure dephasing)
    where γ_↓ = 1/T1, γ_φ = max(0, 1/T2 - 1/(2 T1)).

    Returns normalized fractions (f_X, f_Y, f_Z) summing to 1.
    """
    if T1 <= 0 or T2 <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    gamma_down = 1.0 / T1
    gamma_phi = max(0.0, 1.0 / T2 - 1.0 / (2 * T1))
    # Shape-only (τ cancels when normalizing)
    rX = rY = gamma_down / 4
    rZ = gamma_down / 4 + gamma_phi / 2
    total = rX + rY + rZ
    if total <= 0:
        return (1 / 3, 1 / 3, 1 / 3)
    return (rX / total, rY / total, rZ / total)


def synth_1q_rates(rb_error: float, T1: float, T2: float) -> Dict[str, float]:
    """Pauli-Lindblad rates {X, Y, Z} for a 1Q gate.

    The physical Magnus-M1 rates from T1/T2 give a characteristic {X, Y, Z} fraction
    (see decoherence_fractions). We scale to the RB-calibrated total ε_RB:

        λ_P = f_P × ε_RB     for P ∈ {X, Y, Z}

    RB error absorbs: decoherence (captured by fractions), control miscalibration
    (spread uniformly across P), leakage (Z-like), crosstalk. Distributing by the
    decoherence fraction biases Z > X, Y (typical because T2 < 2T1), matching
    observed 1Q error spectra.
    """
    if rb_error <= 0:
        return {}
    f_X, f_Y, f_Z = decoherence_fractions(T1, T2)
    return {"X": f_X * rb_error, "Y": f_Y * rb_error, "Z": f_Z * rb_error}


def synth_2q_rates(rb_error_2q: float, T1a: float, T2a: float, T1b: float, T2b: float,
                   omega_zz_hz: float, gate_time_s: float) -> Dict[str, float]:
    """Pauli-Lindblad rates for a 2Q gate (cz/ecr) across 15 non-identity Pauli strings.

    The 2Q gate's total RB error ε_RB_2Q is distributed across Pauli-Lindblad generators:

      - Individual qubit decoherence (amp damping + dephasing) on qubits a, b:
          λ_{IX}, λ_{IY}, λ_{IZ}  (qubit a)
          λ_{XI}, λ_{YI}, λ_{ZI}  (qubit b)
        distributed using the same decoherence fractions as 1Q.

      - Correlated ZZ from static coupling during gate:
          λ_{ZZ} = |ω_ZZ| × τ_g                  (Sheldon 2016 formula)

      - Residual (control error, crosstalk, miscalibration): uniformly spread across
        the remaining 2-local generators {XX, XY, XZ, YX, YY, YZ, ZX, ZY} so the
        total sum equals ε_RB_2Q.

    This gives a physics-structured channel that reduces to RB calibration in the
    sum but exposes correlated (ZZ) error separately from individual decoherence.
    """
    if rb_error_2q <= 0:
        return {}

    # (1) Static ZZ coherent error
    lambda_zz = min(abs(omega_zz_hz) * gate_time_s, rb_error_2q * 0.5)

    # (2) Single-qubit decoherence shares on each qubit, weighted by individual error budgets
    fa_X, fa_Y, fa_Z = decoherence_fractions(T1a, T2a)
    fb_X, fb_Y, fb_Z = decoherence_fractions(T1b, T2b)
    # Each qubit's decoherence budget during the 2Q gate is the 1Q idle error over τ_g:
    deco_a = min((1.0 / T1a + max(0, 1.0 / T2a - 1.0 / (2 * T1a)) / 2) * gate_time_s, rb_error_2q * 0.5) if T1a > 0 and T2a > 0 else 0
    deco_b = min((1.0 / T1b + max(0, 1.0 / T2b - 1.0 / (2 * T1b)) / 2) * gate_time_s, rb_error_2q * 0.5) if T1b > 0 and T2b > 0 else 0
    # Cap individual decoherence to leave headroom for ZZ and residual
    deco_cap = max(0, rb_error_2q - lambda_zz) / 2
    deco_a = min(deco_a, deco_cap)
    deco_b = min(deco_b, deco_cap)

    rates: Dict[str, float] = {
        "IX": fa_X * deco_a, "IY": fa_Y * deco_a, "IZ": fa_Z * deco_a,
        "XI": fb_X * deco_b, "YI": fb_Y * deco_b, "ZI": fb_Z * deco_b,
        "ZZ": lambda_zz,
    }

    # (3) Residual budget spread across remaining 2-local generators
    physics_sum = sum(rates.values())
    residual = max(0.0, rb_error_2q - physics_sum)
    residual_generators = ["XX", "XY", "XZ", "YX", "YY", "YZ", "ZX", "ZY"]
    per_gen = residual / len(residual_generators)
    for g in residual_generators:
        rates[g] = per_gen

    return rates


# =============================================================================
# Noise model construction
# =============================================================================

def _pauli_lindblad_error(rates: Dict[str, float], num_qubits: int):
    """Build a Pauli channel QuantumError from {pauli_string: rate}.

    First-order Pauli-Lindblad channel (λ_k small):
        Λ(ρ) = (1 - Σ λ_k) ρ + Σ λ_k P_k ρ P_k†

    We construct via ``pauli_error`` with explicit identity weight `1 - Σ λ_k`
    to guarantee the probabilities sum to exactly 1 (avoiding the floating-point
    drift observed with ``PauliLindbladError.to_quantum_error()`` which caused
    Qiskit Aer to silently skip un-normalized errors at specific noise scales,
    producing non-monotonic fidelity curves).
    """
    from qiskit_aer.noise import pauli_error

    pauli_probs = []
    total = 0.0
    for pauli_str, rate in rates.items():
        if rate <= 0 or not np.isfinite(rate):
            continue
        if len(pauli_str) != num_qubits:
            continue
        pauli_probs.append((pauli_str, float(rate)))
        total += float(rate)
    if not pauli_probs:
        return None
    # Cap at 1 (total should be << 1 in our regime; safety floor against pathology)
    if total >= 1.0:
        # Normalize (keeps channel CPTP when rates unphysically large)
        scale = 0.999999 / total
        pauli_probs = [(p, r * scale) for p, r in pauli_probs]
        total = sum(r for _, r in pauli_probs)
    identity_prob = 1.0 - total
    ops = [("I" * num_qubits, identity_prob)] + pauli_probs
    return pauli_error(ops)


def _add_readout_errors(noise_model, qp_map: dict, phys_to_virt: dict, ro_scale: float):
    """Attach calibrated readout confusion matrices to each qubit (unchanged from legacy)."""
    from qiskit_aer.noise import ReadoutError

    for phys_q, virt_q in phys_to_virt.items():
        qp = qp_map.get(phys_q)
        if qp is None:
            continue
        p0g1 = qp.get("prob_meas0_prep1")
        p1g0 = qp.get("prob_meas1_prep0")
        if p0g1 is not None and p1g0 is not None:
            p0g1_s = min(p0g1 * ro_scale, 0.5)
            p1g0_s = min(p1g0 * ro_scale, 0.5)
            try:
                noise_model.add_readout_error(
                    ReadoutError([[1 - p1g0_s, p1g0_s], [p0g1_s, 1 - p0g1_s]]), [virt_q])
            except (ValueError, RuntimeError):
                pass
        elif "readout_error" in qp:
            err = min(qp["readout_error"] * ro_scale, 0.5)
            if err > 0:
                try:
                    noise_model.add_readout_error(
                        ReadoutError([[1 - err / 2, err / 2], [err / 2, 1 - err / 2]]), [virt_q])
                except (ValueError, RuntimeError):
                    pass


def build_lindblad_noise_model(backend_name: str, physical_qubits: List[int],
                                gate_pulse_map: Optional[dict] = None,
                                duration_scale: float = 1.0,
                                noise_scale: float = 1.0):
    """Construct a Qiskit Aer NoiseModel via Lindblad synthesis (Magnus M1).

    Per gate, the channel is built as:
        E_gate = thermal_a ⊗ thermal_b ⊗ ... ⊙ PauliLindbladError(rates)

    where thermal_k is thermal_relaxation_error(T1_k, T2_k, τ_g) and the
    PauliLindbladError carries physics-distributed RB-calibrated residuals
    including the correlated ZZ term for 2Q gates.

    Args:
        backend_name: Static backend name (e.g. "ibm_torino").
        physical_qubits: Subset of physical qubits to simulate. Virtual indices
            0..N-1 map to these in order.
        gate_pulse_map: Optional override of DEFAULT_GATE_PULSE_MAP (pulse shape
            per gate type). Affects τ_g, leakage, readout SNR.
        duration_scale: Multiplicative scaling of gate duration (for duration
            engineering studies).
        noise_scale: Scales all calibrated rates (RB errors, ro errors) and
            reduces T1, T2 to 1/noise_scale. noise_scale=0 returns empty model.

    Returns:
        NoiseModel ready for AerSimulator.
    """
    from qiskit_aer.noise import NoiseModel, thermal_relaxation_error, depolarizing_error

    if noise_scale == 0.0:
        return NoiseModel()

    backend_data = load_static_backend_data(backend_name)
    qubit_properties = backend_data.get("qubit_properties", [])
    gate_properties = backend_data.get("gate_properties", {})
    basis_gates = backend_data.get("basis_gates", [])

    phys_to_virt = {pq: i for i, pq in enumerate(physical_qubits)}
    qp_map = {qp["qubit"]: qp for qp in qubit_properties}
    gpm = {**DEFAULT_GATE_PULSE_MAP, **(gate_pulse_map or {})}
    noise_model = NoiseModel()

    # Cache scaled coherence times per physical qubit
    scaled_T = {}
    for phys_q in physical_qubits:
        qp = qp_map.get(phys_q)
        if qp is None or qp.get("T1") is None or qp.get("T2") is None:
            continue
        t1 = qp["T1"] / noise_scale
        scaled_T[phys_q] = (t1, _clamp_t2(t1, qp["T2"] / noise_scale))

    # ── 1Q gates ──────────────────────────────────────────────
    for phys_q, virt_q in phys_to_virt.items():
        if phys_q not in scaled_T:
            continue
        t1, t2 = scaled_T[phys_q]

        for gate_name in ("sx", "x", "id", "rz"):
            if gate_name not in basis_gates:
                continue
            gate_time_base = _get_gate_time(gate_properties, gate_name, phys_q)
            gate_shape = gpm.get(gate_name, "drag")
            gate_time = gate_time_base * PULSE_CALIBRATED_DURATION.get(gate_shape, 1.0) * duration_scale

            # Decoherence: exact Qiskit thermal_relaxation (Kraus, all orders)
            try:
                thermal = thermal_relaxation_error(t1, t2, gate_time)
            except (ValueError, RuntimeError):
                thermal = None

            # Pauli-Lindblad residual (RB-anchored, physics-distributed)
            if gate_name in ("id", "rz"):
                # No RB calibration for id/rz → thermal only
                if thermal is not None:
                    noise_model.add_quantum_error(thermal, gate_name, [virt_q])
                continue

            calib_err = gate_properties.get(gate_name, {}).get(f"({phys_q},)", {}).get("error", 0)

            # Pulse-shape modifier: non-DRAG shapes add leakage on top of RB
            rb_effective = calib_err
            if gate_shape != "drag":
                leak_delta = max(0.0, pulse_leakage_rate(gate_shape, gate_time)
                                  - pulse_leakage_rate("drag", gate_time_base))
                rb_effective += leak_delta

            rb_effective = min(rb_effective * noise_scale, 0.75)
            rates = synth_1q_rates(rb_effective, t1, t2)
            pl_err = _pauli_lindblad_error(rates, num_qubits=1)

            # Compose thermal + Pauli channel into one QuantumError
            composite = None
            if thermal is not None and pl_err is not None:
                try:
                    composite = thermal.compose(pl_err)
                except Exception:
                    composite = thermal
            elif pl_err is not None:
                composite = pl_err
            elif thermal is not None:
                composite = thermal

            if composite is not None:
                try:
                    noise_model.add_quantum_error(composite, gate_name, [virt_q])
                except (ValueError, RuntimeError):
                    pass

    # ── 2Q gates ──────────────────────────────────────────────
    for err in backend_data.get("errors", []):
        q_from, q_to = err["qubit_from"], err["qubit_to"]
        if q_from not in phys_to_virt or q_to not in phys_to_virt:
            continue
        if q_from not in scaled_T or q_to not in scaled_T:
            continue

        gate = err["gate"]
        shape_scale = PULSE_2Q_ERROR_SCALE.get(gpm.get(gate, "gaussian_square"), 1.0)
        rb_err_2q = min(err["error_rate"] * shape_scale * duration_scale * noise_scale, 0.75)
        if rb_err_2q <= 0:
            continue

        # Gate duration and static ZZ
        gate_time = _get_gate_time(gate_properties, gate, q_from)
        if gate_time is None or gate_time <= 0:
            gate_time = getattr(Config, "DEFAULT_TWO_QUBIT_GATE_TIME", 400e-9)
        gate_time = gate_time * duration_scale

        freq_a = _FREQ_FALLBACK_GHZ.get(q_from, 5.0)
        freq_b = _FREQ_FALLBACK_GHZ.get(q_to, 5.0)
        detuning = abs(freq_a - freq_b)
        omega_zz_hz = 0.0
        if detuning >= 0.01:
            denom = detuning**2 - DEFAULT_ANHARMONICITY_GHZ**2
            if abs(denom) > 1e-6:
                omega_zz_hz = abs(DEFAULT_COUPLING_GHZ**2 * DEFAULT_ANHARMONICITY_GHZ / denom) * 2 * np.pi * 1e9

        T1a, T2a = scaled_T[q_from]
        T1b, T2b = scaled_T[q_to]

        # Decoherence on each qubit during 2Q gate
        try:
            thermal_a = thermal_relaxation_error(T1a, T2a, gate_time)
            thermal_b = thermal_relaxation_error(T1b, T2b, gate_time)
            thermal_2q = thermal_a.expand(thermal_b)
        except (ValueError, RuntimeError):
            thermal_2q = None

        # Physics-distributed Pauli-Lindblad residual
        rates_2q = synth_2q_rates(rb_err_2q, T1a, T2a, T1b, T2b, omega_zz_hz, gate_time)
        pl_err_2q = _pauli_lindblad_error(rates_2q, num_qubits=2)

        # Compose thermal ⊗ thermal + Pauli channel into one QuantumError
        composite = None
        if thermal_2q is not None and pl_err_2q is not None:
            try:
                composite = thermal_2q.compose(pl_err_2q)
            except Exception:
                composite = thermal_2q
        elif pl_err_2q is not None:
            composite = pl_err_2q
        elif thermal_2q is not None:
            composite = thermal_2q
        else:
            # Last resort: isotropic depolarizing
            try:
                composite = depolarizing_error(rb_err_2q, 2)
            except (ValueError, RuntimeError):
                composite = None

        if composite is not None:
            try:
                noise_model.add_quantum_error(
                    composite, gate, [phys_to_virt[q_from], phys_to_virt[q_to]])
            except (ValueError, RuntimeError):
                pass

    # ── Readout ───────────────────────────────────────────────
    meas_shape = gpm.get("measure", "square")
    ro_scale = PULSE_READOUT_ERROR_SCALE.get(meas_shape, 1.0) * noise_scale
    _add_readout_errors(noise_model, qp_map, phys_to_virt, ro_scale)

    return noise_model
