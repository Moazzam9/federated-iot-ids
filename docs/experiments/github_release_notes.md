# GitHub Release Notes — v1.0.0

**Repository:** `federated-iot-ids`  
**Release Tag:** `v1.0.0`  
**Release Title:** *Federated Learning for Resource-Aware Intrusion Detection in IoT Networks — Reproducibility Package v1.0.0*

---

## Overview

This is the first public release of the reproducibility package accompanying the research study:

> **Title:** *Federated Learning for Resource-Aware Intrusion Detection in IoT Networks: An Empirical Study of IID and Device-Level Non-IID Data*

The release includes the complete experiment codebase, frozen data split specifications, persisted preprocessing artefacts, generated publication figures and tables, the full research manuscript, and a comprehensive reproducibility guide. The study is **not yet peer-reviewed** and has not been assigned a DOI or journal/conference designation.

---

## What Is Included

| Category | Contents |
| :--- | :--- |
| **Manuscript** | `manuscript/manuscript.md` — full single-file research manuscript (628 lines) |
| **Source code** | `src/` — `SmallMLP` model, FedAvg coordinator, N-BaIoT data loader, FL partitioning utilities |
| **Experiment scripts** | `experiments/scripts/` — standalone execution scripts for all four conditions plus analysis |
| **Configuration** | `configs/experiment.yaml` — complete experiment configuration with seed 42 |
| **Data artefacts** | `data/processed/splits/` — frozen split specification; `data/processed/preprocessing/` — persisted `StandardScaler` |
| **Publication figures** | `figures/` and `results/processed/publication/figures/` — 6 publication-ready PNG figures |
| **Publication tables** | `results/processed/publication/tables/` — 6 CSV result tables |
| **Documentation** | `docs/experiments/` — reproducibility guide, scientific consistency audit, literature audit, conclusion validation, limitations validation, final publication package manifest |
| **Tests** | `tests/` — 105-test pytest suite covering model, partitioning, FL logic, reproducibility artefacts |
| **Requirements** | `requirements.txt` — verified Python 3.12 dependency list |

> [!IMPORTANT]
> **Raw execution JSON logs** (`results/raw/`) are tracked in the repository. The dataset itself (N-BaIoT) is **not** included and must be downloaded independently. See the [Reproducibility Guide](reproducibility.md) for dataset acquisition instructions.

---

## Experimental Scope

The release captures a controlled empirical comparison of four training conditions on the N-BaIoT botnet intrusion detection dataset (9 IoT device types, 7,062,606 total observations, 115 continuous features):

| Condition | Test Set | F1-Score | ROC-AUC |
| :--- | :--- | :---: | :---: |
| Centralized Baseline | Global holdout (1,059,388 rows) | 0.9971 | 0.9969 |
| Controlled IID FedAvg (Round 3) | Global holdout (1,059,388 rows) | 0.9994 | 0.9999 |
| Device-Level Non-IID FedAvg (Round 3) | Global holdout (1,059,388 rows) | 0.9757 | 0.9924 |
| Local-Only Baseline (Macro Avg) | Device subsets (per-device) | 0.9714 | 0.9944 |

- **Model:** `SmallMLP` — 4-layer feed-forward network with 9,537 parameters (`115 → 64 → 32 → 1`)
- **Framework:** Vanilla FedAvg with sample-count weighted aggregation
- **Rounds / Epochs:** 3 communication rounds, 1 local epoch per round, 9 clients (full participation)
- **Random seed:** 42 throughout
- **Hardware:** Single-host CPU simulation (Intel Core i7-6820HQ @ 2.70 GHz)

---

## Key Reproducibility Information

1. **Environment:** Python 3.12, PyTorch (CPU), see `requirements.txt` for full dependency list.
2. **Dataset:** N-BaIoT dataset must be downloaded from the UCI ML Repository or its original source. The reproducibility guide specifies the expected directory structure and row counts.
3. **Frozen splits:** `data/processed/splits/` contains the frozen train/validation/test split specifications. Re-running `create_nbaiot_split.py` with seed 42 will reproduce identical splits.
4. **Persisted scaler:** `data/processed/preprocessing/standard_scaler.pkl` is included. Re-fitting with `fit_training_scaler.py` on the original training split will reproduce an equivalent scaler.
5. **Full instructions:** See [`docs/experiments/reproducibility.md`](reproducibility.md).

---

## Important Scope Limitations

Readers and users should be aware of the following documented scope boundaries:

1. **Simulated clients:** The 9 federated clients are logical participants executing on a single host CPU, not physical edge devices.
2. **Pre-fitted scaler:** `StandardScaler` was fitted globally on pooled training data prior to federated partitioning; this does not reflect a fully federated preprocessing pipeline.
3. **Communication accounting:** Reported payload figures (38,148 bytes/transfer; 2,059,992 bytes cumulative) are calculated from model state dict sizes, not physical network socket traffic.
4. **Resource measurements:** RSS memory and wall-clock times reflect host-process execution; no energy, power, or battery consumption was measured.
5. **No privacy evaluation:** Vanilla FedAvg avoids raw data centralization but does not incorporate Differential Privacy, Secure Aggregation, or robustness against adversarial participants.
6. **Single seed, 3 rounds:** Findings are based on a single random seed (42) and a 3-round federated training horizon.

---

## Repository Structure

```text
federated-iot-ids/
├── configs/                    # Experiment configuration YAMLs
├── data/processed/             # Frozen split specs and persisted scaler
├── docs/experiments/           # Reproducibility guide and audit reports
├── docs/methodology/           # Experimental setup documentation
├── experiments/scripts/        # Experiment execution and analysis scripts
├── figures/                    # Top-level publication figures (PNG)
├── manuscript/                 # Full research manuscript
├── results/processed/publication/  # Generated tables (CSV) and figures (PNG)
├── results/raw/                # Raw experiment execution JSON logs
├── src/                        # Core Python package
├── tests/                      # Pytest test suite (105 tests)
├── CITATION.cff                # Software citation metadata
├── requirements.txt            # Python dependency list
└── README.md                   # Repository overview
```

---

## Dataset

This study uses the **N-BaIoT** dataset (Meidan et al., 2018):

> Y. Meidan et al., "N-BaIoT — Network-based Detection of IoT Botnet Attacks Using Deep Autoencoders," *IEEE Pervasive Computing*, vol. 17, no. 3, pp. 12–22, 2018.

The dataset is publicly available from the UCI Machine Learning Repository. It is **not redistributed** in this repository.

---

## Citation

If you use this codebase or reproducibility package, please cite using the metadata in [`CITATION.cff`](../../CITATION.cff) at the repository root.

---

## Automated Test Suite

```powershell
# From repository root, with virtual environment activated:
pytest --basetemp=build/pytest_tmp -q
# Expected: 105 passed
```
