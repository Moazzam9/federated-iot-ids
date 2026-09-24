# Federated Learning for Resource-Aware Intrusion Detection in IoT Networks: An Empirical Study of IID and Device-Level Non-IID Data

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-105%20passed-green.svg)](tests/)
[![License](https://img.shields.io/badge/license-TBD-lightgrey.svg)](#license)

> **Note:** This manuscript is a pre-print and has not yet been peer-reviewed or published. No DOI has been assigned.

---

## Study Overview

This repository contains the complete reproducibility package for a controlled empirical study comparing four training topologies for binary botnet intrusion detection on the **N-BaIoT** dataset.

**Research questions addressed:**

1. How does federated training under a natural device-level Non-IID partition compare to centralized training in detection performance?
2. Does a controlled class-stratified IID partition allow FedAvg to match centralized performance?
3. What are the performance characteristics of fully isolated Local-Only per-device models?
4. What are the communication payload and host-side resource characteristics of FedAvg under these partitions?

**Four experimental conditions:**

| Condition | Test Scope | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: |
| Centralized Baseline | Global holdout (1,059,388 rows) | 0.9971 | 0.9969 |
| Controlled IID FedAvg (Round 3) | Global holdout (1,059,388 rows) | 0.9994 | 0.9999 |
| Device-Level Non-IID FedAvg (Round 3) | Global holdout (1,059,388 rows) | 0.9757 | 0.9924 |
| Local-Only Baseline (Macro Avg) | Device subsets (per-device) | 0.9714 | 0.9944 |

---

## Dataset

The study uses the publicly available **N-BaIoT** dataset (Meidan et al., 2018):

- **9 commercial IoT device types** (e.g., Danmini Doorbell, Ennio Doorbell, Philips B120N/10 Baby Monitor, etc.)
- **7,062,606 total observations**, 115 continuous network traffic features
- **Binary classification target:** benign traffic vs. Mirai / Gafgyt botnet attack traffic
- Dataset is **not redistributed** in this repository and must be downloaded separately.

See [`docs/experiments/reproducibility.md`](docs/experiments/reproducibility.md) for dataset acquisition and directory setup instructions.

---

## Model and Framework

- **Model:** `SmallMLP` — 4-layer feed-forward neural network: `115 → Linear(64) → ReLU → Linear(32) → ReLU → Linear(1) → Sigmoid`; **9,537 parameters** total.
- **FL framework:** Vanilla Federated Averaging (FedAvg) with sample-count weighted aggregation; 9 clients, full participation every round.
- **Training horizon:** 3 communication rounds, 1 local epoch per round (federated); 3 epochs (centralized).
- **Random seed:** 42 throughout all experiments.
- **Hardware:** Single-host CPU simulation (Intel Core i7-6820HQ @ 2.70 GHz, ~17 GB RAM). All 9 clients are logical participants on one machine, not physical edge devices.

---

## Quick Start & Reproducibility

For complete step-by-step instructions covering environment setup, dataset acquisition, experiment execution, resource measurement, and results visualization:

👉 **[Reproducibility Guide — docs/experiments/reproducibility.md](docs/experiments/reproducibility.md)**

**Key experiment entry points:**

```powershell
# Centralized baseline
.\.venv\Scripts\python.exe -m experiments.scripts.run_centralized

# Local-only per-device baseline
.\.venv\Scripts\python.exe -m experiments.scripts.run_local_only

# Device-level Non-IID FedAvg (3 rounds)
.\.venv\Scripts\python.exe -m experiments.scripts.run_multiround_federated `
    --partition device_non_iid --rounds 3 --local-epochs 1 `
    --batch-size 256 --learning-rate 0.001 --seed 42 `
    --output-prefix device_non_iid_3round_seed42

# Controlled IID FedAvg (3 rounds)
.\.venv\Scripts\python.exe -m experiments.scripts.run_multiround_federated `
    --partition iid --rounds 3 --local-epochs 1 `
    --batch-size 256 --learning-rate 0.001 --seed 42 `
    --output-prefix iid_3round_seed42
```

> **Important:** Raw experiments are expensive (hours of CPU time per condition). The repository already includes pre-computed raw JSON logs, frozen split specifications, a persisted scaler, and generated figures and tables so that results can be inspected and validated without re-running the full experiments.

---

## Repository Structure

```text
federated-iot-ids/
├── configs/                         # Experiment configuration YAMLs
│   └── experiment.yaml              # Primary config (seed 42, SmallMLP, FedAvg)
├── data/
│   └── processed/
│       ├── splits/                  # Frozen N-BaIoT train/val/test split specifications
│       └── preprocessing/           # Persisted StandardScaler binary (.pkl)
├── docs/
│   ├── experiments/                 # Reproducibility guide, audit reports, release docs
│   └── methodology/                 # Experimental setup and FL framework documentation
├── experiments/
│   └── scripts/                     # Standalone scripts: run_centralized.py,
│                                    #   run_local_only.py, run_multiround_federated.py,
│                                    #   build_publication_results.py, evaluate_final_test_set.py, etc.
├── figures/                         # Top-level publication figures (PNG, 6 files)
├── manuscript/
│   └── manuscript.md                # Full research manuscript (~628 lines)
├── results/
│   ├── processed/publication/       # Generated figures (PNG) and tables (CSV)
│   └── raw/                         # Raw experiment execution JSON logs
├── src/                             # Core Python package
│   ├── data/                        # N-BaIoT data loader and split utilities
│   ├── fl/                          # FedAvg coordinator and partition utilities
│   └── models/                      # SmallMLP model definition
├── tests/                           # Pytest test suite (105 tests)
├── CITATION.cff                     # Software citation metadata
├── requirements.txt                 # Verified Python 3.12 dependency list
└── README.md                        # This file
```

---

## Running the Test Suite

```powershell
# From the repository root with the virtual environment activated:
.\.venv\Scripts\Activate.ps1
pytest --basetemp=build/pytest_tmp -q
```

Expected result: **`105 passed`**

The test suite does **not** require the N-BaIoT dataset to be present. All tests use frozen artefacts or lightweight synthetic inputs.

---

## Scope Limitations

The following scope boundaries are explicitly documented in the manuscript (Section 7):

1. **Simulated clients** — 9 logical participants on one host; not physical edge devices.
2. **Pre-fitted scaler** — `StandardScaler` was fitted globally on pooled training data before federated partitioning; this does not reflect a fully federated preprocessing pipeline.
3. **Communication accounting** — payload figures are calculated from model state dict sizes, not physical socket traffic or transport protocol overhead.
4. **Resource measurements** — RSS memory and wall-clock times are host-process measurements; no energy or battery consumption was measured.
5. **No privacy evaluation** — vanilla FedAvg avoids raw data centralization but does not implement Differential Privacy, Secure Aggregation, or robustness analysis against adversarial participants.
6. **Single seed, 3 rounds** — findings are based on a single random seed (42) and a 3-round federated training horizon; long-run convergence behaviour is unmeasured.

---

## Key Documentation

| Document | Purpose |
| :--- | :--- |
| [`docs/experiments/reproducibility.md`](docs/experiments/reproducibility.md) | Step-by-step reproducibility guide |
| [`docs/experiments/scientific_consistency_audit.md`](docs/experiments/scientific_consistency_audit.md) | 29-area scientific consistency audit report |
| [`docs/experiments/literature_audit.md`](docs/experiments/literature_audit.md) | Primary literature citation audit (15 references) |
| [`docs/experiments/final_publication_package_manifest.md`](docs/experiments/final_publication_package_manifest.md) | Package manifest with verified headline metrics |
| [`docs/experiments/github_release_notes.md`](docs/experiments/github_release_notes.md) | v1.0.0 release notes |
| [`docs/experiments/github_release_checklist.md`](docs/experiments/github_release_checklist.md) | Pre-release checklist |
| [`docs/methodology/experimental_setup.md`](docs/methodology/experimental_setup.md) | Detailed technical setup documentation |
| [`manuscript/manuscript.md`](manuscript/manuscript.md) | Full research manuscript |
| [`CITATION.cff`](CITATION.cff) | Software citation metadata |

---

## License

> **A LICENSE file has not yet been added to this repository.**  
> A license must be selected before the public GitHub Release is created.  
> Common choices for academic research code: MIT License, Apache 2.0, or GNU GPLv3.

---

## Citation

If you use this codebase or reproducibility package, please cite using the metadata in [`CITATION.cff`](CITATION.cff).

The N-BaIoT dataset should be cited as:

> Y. Meidan, M. Bohadana, Y. Mathov, Y. Mirsky, A. Shabtai, D. Breitenbacher, and Y. Elovici, "N-BaIoT — Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders," *IEEE Pervasive Computing*, vol. 17, no. 3, pp. 12–22, Jul.–Sep. 2018.
