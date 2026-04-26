# QuBridge — Reproducibility Artifact

Cached calibration-derived simulation outputs and figure data for the QCE
short paper *"QuBridge: Layer-wise Fidelity Decomposition in Quantum
Computation Pipeline"*.

This artifact reproduces the reported tables and plots from cached
calibration-derived simulation outputs. **Live IBM Quantum hardware access
is not required for reproduction.**

## Layout

```
qubridge-artifact/
├── README.md
├── LICENSE                                    # MIT
├── environment.yml                            # conda env (qiskit 1.3.1, aer 0.16.0)
├── requirements.txt                           # pip equivalent
├── calibration/
│   └── ibm_torino_2026-01-16.json             # cached IBM Torino snapshot
├── data/                                      # raw CSVs
│   ├── table1_pipeline_trace.csv              # Table I, Fig 2(b), Fig 4(b)
│   ├── table2_l2_filter.csv                   # Table II, Fig 3(b)
│   └── table3_encoded_teleportation.csv       # Table III, Fig 5
├── figures/                                   # rendered PDFs (reference)
│   ├── fig2b_pipeline_convergence.pdf
│   ├── fig3b_filter_band.pdf
│   ├── fig4b_pergate_comparison.pdf
│   └── fig5_noise_scaling.pdf
├── notebooks/
│   ├── reproduce_all_en.ipynb                 # one-stop Jupyter notebook (English)
│   ├── reproduce_all_ja.ipynb                 # one-stop Jupyter notebook (Japanese)
│   ├── walkthrough_logical_en.ipynb           # logical-qubit walkthrough (English)
│   └── walkthrough_logical_ja.ipynb           # logical-qubit walkthrough (Japanese)
├── qubridge_logical/                          # vendored deps for walkthrough notebooks
│   ├── __init__.py
│   ├── circuits.py                            # logical/physical teleportation builders + DM fidelity
│   ├── static_backend.py                      # calibration JSON loader
│   ├── qubit_selection.py                     # VF2 layout selection
│   ├── noise_utils.py                         # reduced noise model API
│   └── lindblad_synthesis.py                  # Magnus M1 Pauli-Lindblad noise synthesis
└── scripts/
    ├── reproduce_table1.py                    # Table I summary
    ├── reproduce_table2.py                    # Table II summary
    ├── reproduce_table3.py                    # Table III summary
    ├── plot_figures.py                        # regenerate Figs 2(b)/3(b)/4(b)/5
    └── plot_example.py                        # minimal Python script demo
```

The `reproduce_all_*` notebooks and `scripts/*.py` only read CSVs from `data/`
and have no Qiskit dependency at runtime. The `walkthrough_logical_*`
notebooks construct the [[2,1,1]] logical-teleportation circuit live and use
the vendored `qubridge_logical/` package together with the
`calibration/ibm_torino_2026-01-16.json` snapshot — no upstream source tree is
required.

## Data ↔ paper mapping

| Paper artifact | CSV file | Quick verification |
|---|---|---|
| Table I | `data/table1_pipeline_trace.csv` | `python scripts/reproduce_table1.py` reports per-stage Min / Avg / Max F + band width |
| Table II | `data/table2_l2_filter.csv` | `python scripts/reproduce_table2.py` reports Worst / Best F per filter row |
| Table III | `data/table3_encoded_teleportation.csv` | `python scripts/reproduce_table3.py` pivots on (state, noise_scale) and reports Phys / Log F + acceptance |
| Fig 2(b), 3(b), 4(b), 5 | (same CSVs as above) | `python scripts/plot_figures.py` regenerates all four |

## Reproducing the paper

### Option A — Jupyter notebook (one-stop)

```bash
pip install -r requirements.txt
jupyter notebook notebooks/reproduce_all_en.ipynb     # or _ja.ipynb
```

Run all cells (`Cell → Run All`); each section loads one CSV, prints a
summary table matching the paper, and plots the corresponding figure.

### Option B — Python scripts

```bash
pip install -r requirements.txt

python scripts/reproduce_table1.py        # Table I
python scripts/reproduce_table2.py        # Table II
python scripts/reproduce_table3.py        # Table III
python scripts/plot_figures.py            # regenerate Figs 2(b)/3(b)/4(b)/5 as PNG
```

### Option C — Conda environment

```bash
conda env create -f environment.yml
conda activate qubridge-qce26
python scripts/plot_figures.py
```

### Option D — Logical-teleportation walkthrough (live circuit + DM simulation)

```bash
pip install -r requirements.txt
jupyter notebook notebooks/walkthrough_logical_en.ipynb     # or _ja.ipynb
```

Reproduces Paper Table V (noise=1.0, |+⟩, logical) live: builds the
six-qubit [[2,1,1]] deferred-measurement circuit, picks a layout on IBM
Torino via VF2, constructs a Magnus-M1 Pauli-Lindblad noise model from the
cached calibration snapshot, runs an Aer density-matrix simulation, and
reports Phys F = 0.9849 / Log F = 0.9769 / Accept = 0.9263. Imports come
from the vendored `qubridge_logical/` package; calibration data from
`calibration/ibm_torino_2026-01-16.json`.

## Simulation environment

| Item | Value |
|---|---|
| Backend calibration | IBM Torino, snapshot 2026-01-16 (see `calibration/`) |
| Qiskit | 1.3.1 |
| Qiskit Aer | 0.16.0 |
| Python | >= 3.10 |
| Simulation method | density matrix |
| Transpiler optimization | level 3 |
| VF2 tie-breaking seed | 42 |
| Primary metric | state fidelity F = ⟨ψ|ρ|ψ⟩ |

Density-matrix simulation is deterministic; per-configuration repeats in
the CSVs ensure logging-schema consistency rather than statistical noise
estimates.

## Citation

```
(BibTeX entry to be added on acceptance.)
```

## License

Released under the MIT License — see [LICENSE](LICENSE).
