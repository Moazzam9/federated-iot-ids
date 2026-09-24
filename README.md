# Federated Learning for Resource-Aware Intrusion Detection in IoT Networks: An Empirical Study of IID and Device-Level Non-IID Data

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![PyTorch 2.14](https://img.shields.io/badge/PyTorch-2.14-orange.svg)](https://pytorch.org/)
[![Tests](https://img.shields.io/badge/tests-100%20passed-green.svg)]()

Official research repository and reproducibility package for **the study**:  
*"Federated Learning for Resource-Aware Intrusion Detection in IoT Networks: An Empirical Study of IID and Device-Level Non-IID Data"*.

---

## Quick Start & Reproducibility

For step-by-step instructions on environment setup, dataset acquisition, experiment execution, resource measurement, and results visualization, please refer to the primary reproducibility guide:

👉 **[Reproducibility Guide (docs/experiments/reproducibility.md)](docs/experiments/reproducibility.md)**

---

## Study Summary

This paper presents a controlled empirical study comparing four training conditions for binary botnet intrusion detection (Mirai and Gafgyt attack vectors) on the N-BaIoT dataset across 9 commercial IoT device types (7,062,606 total observations, 115 continuous features):

1. **Centralized Baseline:** Model trained on pooled global training data (F1: **0.9971**).
2. **Controlled IID FedAvg:** Federated Averaging across class-stratified IID partitions (F1: **0.9994**).
3. **Device-Level Non-IID FedAvg:** Federated Averaging across natural device partitions (F1: **0.9757**).
4. **Local-Only Baseline:** Isolated per-device client models (Macro F1: **0.9714**).

All experiments evaluate a lightweight 9,537-parameter feed-forward neural network (`SmallMLP`) implemented in PyTorch under CPU-only execution.

---

## Repository Structure

```text
federated-iot-ids/
├── configs/                # Experiment configuration YAMLs
├── data/
│   ├── processed/splits/   # Frozen N-BaIoT split specifications
│   └── processed/preprocessing/ # Persisted StandardScaler binary
├── docs/
│   ├── experiments/        # Reproducibility guide & scientific audit reports
│   └── methodology/        # Experimental setup and FL framework documentation
├── experiments/
│   └── scripts/            # Standalone execution scripts for experiments & analysis
├── manuscript/             # Full research manuscript (manuscript.md)
├── results/
│   └── processed/publication/ # Generated publication tables and figures
├── src/                    # Core Python package (data loaders, models, FL coordinator)
├── tests/                  # Pytest unit and integration test suite
├── .gitignore              # Repository git ignore specifications
├── requirements.txt        # Verified Python dependency specifications
└── README.md               # Repository overview (this file)
```

---

## Running Validation Tests

To run the automated unit and integration test suite:

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1

# Execute test suite
pytest --basetemp=build/pytest_tmp -q
```

Expected result: `100 passed`.

---

## License & Citation

This repository is part of the research project. Please see the [Manuscript](manuscript/manuscript.md) for full citation metadata and bibliographic references.
