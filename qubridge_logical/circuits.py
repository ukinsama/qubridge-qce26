"""Logical / physical teleportation circuit builders and density-matrix fidelity.

Vendored subset of ``services/mode3_computation.py`` (upstream QuBridge) trimmed
to the functions used by ``notebooks/walkthrough_logical_*.ipynb``:

  - create_deferred_logical_teleportation_circuit
  - create_logical_teleportation_circuit (Method 2 with mid-circuit measurement)
  - create_deferred_teleportation_circuit (3-qubit physical, deferred form)
  - create_physical_teleportation_circuit (3-qubit physical, mid-circuit form)
  - compute_dm_fidelity_logical
  - compute_dm_fidelity_physical

UI helpers (matplotlib bar charts), noise sweeps and ``compare_*`` orchestration
are intentionally omitted; the walkthrough constructs those plots inline. The
helper ``create_deferred_teleportation_circuit`` was originally in
``utils/circuit.py`` and is inlined here so this module has no upstream
dependencies beyond ``.static_backend`` and ``.noise_utils``.
"""
from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


# =============================================================================
# Circuit builders
# =============================================================================

def create_logical_teleportation_circuit(theta: float, phi: float):
    """[[2,1,1]] logical teleportation circuit (CX_L + H_L decode-encode).

    Protocol:
      1. Alice prepares |ψ_L⟩ = α|00⟩+β|11⟩ via ry(θ)·rz(φ)·CX(0,1)
      2. Mediator-Bob logical Bell pair |Φ+_L⟩ = (|0000⟩+|1111⟩)/√2
      3. Logical Bell measurement on (Alice, Mediator):
         - CX_L (transversal): CX(A_0,M_0); CX(A_1,M_1)
         - H_L (decode-encode): CX(A_0,A_1); H(A_0); CX(A_0,A_1)
         - Z-basis measurement on Alice, Mediator
      4. Bob logical Pauli correction: X_L^{m_M} Z_L^{m_A} (X_L=X⊗X, Z_L=Z⊗I)
    """
    from qiskit import QuantumCircuit, QuantumRegister, ClassicalRegister

    qa = QuantumRegister(2, "alice")
    qm = QuantumRegister(2, "med")
    qb = QuantumRegister(2, "bob")

    cr_meas_a = ClassicalRegister(2, "meas_a")
    cr_meas_m = ClassicalRegister(2, "meas_m")
    cr_bob = ClassicalRegister(2, "c_bob")
    qc = QuantumCircuit(qa, qm, qb, cr_meas_a, cr_meas_m, cr_bob)

    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.cx(0, 1)
    qc.barrier(label="Alice encoded")

    qc.h(2)
    qc.cx(2, 3)
    qc.cx(2, 4)
    qc.cx(3, 5)
    qc.barrier(label="Bell pair")

    qc.cx(0, 2)
    qc.cx(1, 3)
    qc.barrier(label="CX_L")
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(0, 1)
    qc.barrier(label="H_L decode-encode")
    qc.measure(0, cr_meas_a[0])
    qc.measure(1, cr_meas_a[1])
    qc.measure(2, cr_meas_m[0])
    qc.measure(3, cr_meas_m[1])
    qc.barrier(label="Measure A, M")

    with qc.if_test((cr_meas_m[0], 1)):
        qc.x(4)
        qc.x(5)
    with qc.if_test((cr_meas_a[0], 1)):
        qc.z(4)
    qc.barrier(label="Corrected")

    qc.measure(4, cr_bob[0])
    qc.measure(5, cr_bob[1])

    return qc


def create_deferred_logical_teleportation_circuit(theta: float, phi: float,
                                                   add_barriers: bool = True):
    """Deferred-measurement form of the [[2,1,1]] logical teleportation circuit.

    Equivalent to ``create_logical_teleportation_circuit`` with ``if_test``
    Pauli corrections replaced by quantum-controlled gates so the circuit is
    compatible with ``save_density_matrix()`` for true (phase-aware) state
    fidelity calculations.

    Deferred substitution:
      measure(qm0) + if(1): X(qb0); X(qb1)  ≡  CX(qm0, qb0); CX(qm0, qb1)
      measure(qa0) + if(1): Z(qb0)          ≡  CZ(qa0, qb0)

    Stages (separated by barriers when ``add_barriers=True``):
      1. Alice encoding         — prepare |ψ_L⟩ = α|00⟩+β|11⟩ on (qa0, qa1)
      2. Mediator-Bob Bell pair — entangle med (qm0, qm1) with bob (qb0, qb1)
      3. CX_L (transversal)     — Alice → Mediator coupling
      4. H_L (decode-encode)    — non-transversal Hadamard on Alice
      5. Deferred correction    — CX/CZ replacing post-measurement Paulis

    Args:
        theta, phi: input Bloch angles.
        add_barriers: if True (default), insert labelled barriers between the
            five stages. Barriers are visualization-only markers but they also
            constrain ``optimization_level=3`` transpilation; the fidelity is
            unchanged within rounding for the Table V (noise=1.0, |+⟩) point.
    """
    from qiskit import QuantumCircuit, QuantumRegister

    qa = QuantumRegister(2, "alice")
    qm = QuantumRegister(2, "med")
    qb = QuantumRegister(2, "bob")
    qc = QuantumCircuit(qa, qm, qb)

    # 1. Alice encoding: |ψ_L⟩ = α|00⟩ + β|11⟩
    qc.ry(theta, 0)
    qc.rz(phi, 0)
    qc.cx(0, 1)
    if add_barriers:
        qc.barrier(label="Alice encoded")

    # 2. Mediator-Bob logical Bell pair: (|0000⟩+|1111⟩)/√2
    qc.h(2)
    qc.cx(2, 3)
    qc.cx(2, 4)
    qc.cx(3, 5)
    if add_barriers:
        qc.barrier(label="Bell pair")

    # 3. CX_L (transversal): Alice → Mediator
    qc.cx(0, 2)
    qc.cx(1, 3)
    if add_barriers:
        qc.barrier(label="CX_L")

    # 4. H_L on Alice via decode-encode (Eastin–Knill: not transversal)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(0, 1)
    if add_barriers:
        qc.barrier(label="H_L decode-encode")

    # 5. Deferred Pauli correction on Bob (X_L^{m_M} then Z_L^{m_A})
    qc.cx(2, 4)  # was: measure(qm0) + if(1): X(qb0)
    qc.cx(2, 5)  # was: measure(qm0) + if(1): X(qb1)
    qc.cz(0, 4)  # was: measure(qa0) + if(1): Z(qb0)

    return qc


def create_physical_teleportation_circuit(theta_rad: float, phi_rad: float):
    """3-qubit physical teleportation circuit (with mid-circuit measurement)."""
    from qiskit import QuantumCircuit, ClassicalRegister

    cr_meas = ClassicalRegister(2, 'meas')
    cr_bob_phys = ClassicalRegister(1, 'bob')
    qc = QuantumCircuit(3, name='physical_teleport')
    qc.add_register(cr_meas)
    qc.add_register(cr_bob_phys)

    qc.ry(theta_rad, 0)
    qc.rz(phi_rad, 0)

    qc.h(1)
    qc.cx(1, 2)
    qc.barrier()

    qc.cx(0, 1)
    qc.h(0)
    qc.measure(0, cr_meas[0])
    qc.measure(1, cr_meas[1])
    qc.barrier()

    with qc.if_test((cr_meas[1], 1)):
        qc.x(2)
    with qc.if_test((cr_meas[0], 1)):
        qc.z(2)

    qc.measure(2, cr_bob_phys[0])

    return qc


def create_deferred_teleportation_circuit(theta_angle: float, phi_angle: float):
    """Deferred-measurement form of the 3-qubit physical teleportation circuit.

    Vendored from upstream ``utils/circuit.py``. Uses CNOT/CZ instead of
    mid-circuit measurement so the circuit is compatible with
    ``save_density_matrix()``.

    Equivalences:
        measure(q0) + if(1): Z(q2)  ≡  CZ(q0, q2)
        measure(q1) + if(1): X(q2)  ≡  CNOT(q1, q2)
    """
    from qiskit import QuantumCircuit

    qc = QuantumCircuit(3)
    qc.ry(theta_angle, 0)
    qc.rz(phi_angle, 0)
    qc.h(1)
    qc.cx(1, 2)
    qc.cx(0, 1)
    qc.h(0)
    qc.cx(1, 2)
    qc.cz(0, 2)
    return qc


# =============================================================================
# Density-matrix fidelity (true state fidelity, φ-sensitive)
# =============================================================================

def ideal_bloch_probs(theta_rad: float):
    """Ideal Bloch sphere measurement probabilities (cos²(θ/2), sin²(θ/2))."""
    return float(np.cos(theta_rad / 2) ** 2), float(np.sin(theta_rad / 2) ** 2)


def _ideal_logical_bob_state(theta_rad: float, phi_rad: float):
    """Ideal Bob logical state |ψ_L⟩ = cos(θ/2)|00⟩ + e^{iφ}·sin(θ/2)|11⟩."""
    from qiskit.quantum_info import Statevector

    c = np.cos(theta_rad / 2)
    s = np.sin(theta_rad / 2)
    return Statevector([c, 0, 0, np.exp(1j * phi_rad) * s])


def _ideal_physical_bob_state(theta_rad: float, phi_rad: float):
    """Ideal Bob physical state |ψ⟩ = cos(θ/2)|0⟩ + e^{iφ}·sin(θ/2)|1⟩."""
    from qiskit.quantum_info import Statevector

    c = np.cos(theta_rad / 2)
    s = np.sin(theta_rad / 2)
    return Statevector([c, np.exp(1j * phi_rad) * s])


def _readout_error_for(backend_name: str, physical_qubit: int,
                       gate_pulse_map: dict | None = None) -> float:
    """Per-qubit readout error rate (with optional pulse-shape scaling)."""
    try:
        from .static_backend import load_static_backend_data
        data = load_static_backend_data(backend_name)
        base = 0.0
        for qp in data.get('qubit_properties', []):
            if qp.get('qubit') == physical_qubit:
                base = float(qp.get('readout_error', 0.0))
                break
    except Exception:
        base = 0.0

    if gate_pulse_map is None or base <= 0:
        return base

    try:
        from .noise_utils import (
            PULSE_READOUT_ERROR_SCALE, DEFAULT_GATE_PULSE_MAP,
        )
        meas_shape = gate_pulse_map.get(
            "measure", DEFAULT_GATE_PULSE_MAP.get("measure", "square"))
        scale = PULSE_READOUT_ERROR_SCALE.get(meas_shape, 1.0)
        return min(base * scale, 0.5)
    except Exception:
        return base


def _apply_readout_error_physical(bob_dm, p_avg: float):
    """Apply X+Z Pauli mixing to a 1-qubit DM to model misidentified corrections."""
    from qiskit.quantum_info import DensityMatrix
    if p_avg <= 0:
        return bob_dm
    p = min(p_avg, 0.5)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    rho = np.array(bob_dm.data)
    rho = (1 - p) * rho + (p / 2) * (Z @ rho @ Z) + (p / 2) * (X @ rho @ X)
    return DensityMatrix(rho)


def _apply_readout_error_logical(bob_dm, p_avg: float):
    """Apply X_L+Z_L Pauli mixing to a 2-qubit logical Bob DM."""
    from qiskit.quantum_info import DensityMatrix
    if p_avg <= 0:
        return bob_dm
    p = min(p_avg, 0.5)
    Iop = np.eye(2, dtype=complex)
    X = np.array([[0, 1], [1, 0]], dtype=complex)
    Z = np.array([[1, 0], [0, -1]], dtype=complex)
    Z_L = np.kron(Z, Iop)
    X_L = np.kron(X, X)
    rho = np.array(bob_dm.data)
    rho = (1 - p) * rho + (p / 2) * (Z_L @ rho @ Z_L) + (p / 2) * (X_L @ rho @ X_L)
    return DensityMatrix(rho)


def _transpile_for_noise(qc, noise_model, coupling_map=None, initial_layout=None):
    """Transpile to a basis matching the noise model's noise_instructions.

    The noise model carries a ``cz`` 2Q error but ``noise_model.basis_gates``
    advertises both ``cx`` and ``cz``. Transpiling against ``basis_gates``
    leaves stray ``cx`` gates untouched by the noise pass; we instead use the
    actual ``noise_instructions`` (plus virtual-Z) so every emitted gate
    carries the calibrated channel.
    """
    from qiskit import transpile as qk_transpile

    if noise_model is None:
        return qc
    noise_gates = set(noise_model.noise_instructions) - {'measure', 'reset', 'barrier'}
    if not noise_gates:
        return qc
    basis = list(noise_gates | {'rz'})
    return qk_transpile(
        qc,
        basis_gates=basis,
        coupling_map=coupling_map,
        initial_layout=initial_layout,
        optimization_level=3,
        seed_transpiler=42,
    )


def compute_dm_fidelity_logical(theta_rad: float, phi_rad: float, noise_model=None,
                                 backend_name=None, selected_qubits=None,
                                 gate_pulse_map: dict | None = None,
                                 noise_scale: float = 1.0) -> dict:
    """True logical fidelity from a deferred-measurement DM simulation.

    Procedure:
      1. Build the deferred logical teleportation circuit
      2. Transpile against a coupling map reduced to the selected physical qubits
         so routing SWAPs stay inside the selected set
      3. Run AerSimulator(method='density_matrix')
      4. Project onto the codeword subspace ({00,11} ⊗ {00,11}) for syndrome
         post-selection
      5. Partial-trace over Alice + Mediator and compare to the ideal logical
         Bob state. Optionally apply post-hoc readout Pauli mixing.
    """
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import state_fidelity, partial_trace, DensityMatrix

    # Build the circuit without barriers for the simulation: barriers would
    # constrain optimization_level=3 transpilation and shift the fidelity by
    # O(1e-3). The user-facing `create_deferred_logical_teleportation_circuit`
    # default has barriers ON for visualization.
    qc = create_deferred_logical_teleportation_circuit(theta_rad, phi_rad,
                                                       add_barriers=False)

    reduced_cm = None
    if backend_name and selected_qubits and len(selected_qubits) >= 6:
        from .static_backend import load_static_backend_data
        full_cm = load_static_backend_data(backend_name).get('coupling_map', [])
        q_to_idx = {q: i for i, q in enumerate(selected_qubits[:6])}
        reduced_cm = [
            [q_to_idx[a], q_to_idx[b]] for a, b in full_cm
            if a in q_to_idx and b in q_to_idx
        ] or None
    qc = _transpile_for_noise(qc, noise_model, coupling_map=reduced_cm,
                               initial_layout=list(range(qc.num_qubits)) if reduced_cm else None)
    qc.save_density_matrix()

    sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    full_dm = sim.run(qc).result().data()['density_matrix']

    try:
        final_layout = qc.layout.final_index_layout() if qc.layout else list(range(qc.num_qubits))
    except Exception:
        final_layout = list(range(qc.num_qubits))

    n_qubits = qc.num_qubits
    phys_to_virt = [0] * n_qubits
    for virt, pos in enumerate(final_layout):
        if pos < n_qubits:
            phys_to_virt[pos] = virt

    dim = 1 << n_qubits
    full_dm_arr = full_dm.data
    valid_mask = np.zeros(dim, dtype=bool)
    for idx in range(dim):
        bits_phys = [(idx >> k) & 1 for k in range(n_qubits)]
        bits_virt = [0] * n_qubits
        for k in range(n_qubits):
            bits_virt[phys_to_virt[k]] = bits_phys[k]
        if bits_virt[0] == bits_virt[1] and bits_virt[2] == bits_virt[3]:
            valid_mask[idx] = True

    rho_post = np.zeros_like(full_dm_arr)
    rho_post[np.ix_(valid_mask, valid_mask)] = full_dm_arr[np.ix_(valid_mask, valid_mask)]
    p_valid_syndrome = float(np.real(np.trace(rho_post)))
    if p_valid_syndrome < 1e-12:
        from qiskit.quantum_info import DensityMatrix as _DM
        bob_dm = _DM(np.zeros((4, 4), dtype=complex))
        syndrome_rejected = 1.0
    else:
        rho_post /= p_valid_syndrome
        syndrome_rejected = max(0.0, 1.0 - p_valid_syndrome)
        to_trace = [final_layout[i] for i in range(4)]
        bob_dm = partial_trace(DensityMatrix(rho_post), to_trace)

    has_noise = noise_model is not None and len(noise_model.noise_instructions) > 0
    if has_noise and backend_name and selected_qubits and len(selected_qubits) >= 4:
        p_a = _readout_error_for(backend_name, selected_qubits[0], gate_pulse_map) * noise_scale
        p_m = _readout_error_for(backend_name, selected_qubits[2], gate_pulse_map) * noise_scale
        bob_dm = _apply_readout_error_logical(bob_dm, min((p_a + p_m) / 2, 0.5))

    bob_arr = bob_dm.data
    p_00 = float(bob_arr[0, 0].real)
    p_11 = float(bob_arr[3, 3].real)
    p_codespace = p_00 + p_11
    error_rate = max(0.0, 1.0 - (1.0 - syndrome_rejected) * p_codespace)

    if p_codespace > 1e-10:
        proj = np.zeros((4, 4), dtype=complex)
        proj[0, 0] = 1
        proj[3, 3] = 1
        bob_ps = (proj @ bob_arr @ proj) / p_codespace
        ideal = DensityMatrix(_ideal_logical_bob_state(theta_rad, phi_rad))
        fidelity = float(state_fidelity(ideal, DensityMatrix(bob_ps)))
        p_meas_0 = p_00 / p_codespace
        p_meas_1 = p_11 / p_codespace
    else:
        fidelity = 0.0
        p_meas_0 = 0.0
        p_meas_1 = 0.0

    p_ideal_0, p_ideal_1 = ideal_bloch_probs(theta_rad)

    return {
        'fidelity': round(fidelity, 6),
        'error_detection_rate': round(error_rate, 4),
        'p_measured_0': round(p_meas_0, 4),
        'p_measured_1': round(p_meas_1, 4),
        'p_ideal_0': round(p_ideal_0, 4),
        'p_ideal_1': round(p_ideal_1, 4),
    }


def compute_dm_fidelity_physical(theta_rad: float, phi_rad: float, noise_model=None,
                                  backend_name=None, selected_qubits=None,
                                  gate_pulse_map: dict | None = None,
                                  noise_scale: float = 1.0) -> dict:
    """True physical teleportation fidelity from a deferred-measurement DM sim."""
    from qiskit_aer import AerSimulator
    from qiskit.quantum_info import state_fidelity, partial_trace, DensityMatrix

    qc = create_deferred_teleportation_circuit(theta_rad, phi_rad)
    qc = _transpile_for_noise(qc, noise_model)
    qc.save_density_matrix()

    sim = AerSimulator(method='density_matrix', noise_model=noise_model)
    full_dm = sim.run(qc).result().data()['density_matrix']

    bob_dm = partial_trace(full_dm, [0, 1])

    has_noise = noise_model is not None and len(noise_model.noise_instructions) > 0
    if has_noise and backend_name and selected_qubits and len(selected_qubits) >= 2:
        p0 = _readout_error_for(backend_name, selected_qubits[0], gate_pulse_map) * noise_scale
        p1 = _readout_error_for(backend_name, selected_qubits[1], gate_pulse_map) * noise_scale
        bob_dm = _apply_readout_error_physical(bob_dm, min((p0 + p1) / 2, 0.5))

    bob_arr = bob_dm.data
    ideal = DensityMatrix(_ideal_physical_bob_state(theta_rad, phi_rad))
    fidelity = float(state_fidelity(ideal, bob_dm))

    return {
        'fidelity': round(fidelity, 6),
        'p_measured_0': round(float(bob_arr[0, 0].real), 4),
        'p_measured_1': round(float(bob_arr[1, 1].real), 4),
    }
