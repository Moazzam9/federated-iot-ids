<div align="center">

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22941258.svg)](https://doi.org/10.5281/zenodo.22941258)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-CPU-ee4c2c.svg)](https://pytorch.org/)

</div>

# Federated Learning for Resource-Aware Intrusion Detection in IoT Networks

A reproducible empirical study of IID and device-level Non-IID federated learning on the N-BaIoT botnet detection benchmark.

**Author:** Moazzam Azam — moazzamkk13@gmail.com — [ORCID: 0009-0001-6145-0473](https://orcid.org/0009-0001-6145-0473)

---

## Overview

The proliferation of Internet of Things (IoT) devices has significantly expanded the attack surface for network intrusion and botnet-driven malware campaigns. Centralised machine learning approaches for intrusion detection require transmitting raw, often privacy-sensitive, network traffic to a single aggregation server, which is architecturally infeasible at IoT scale. Federated learning (FL) addresses this by training models collaboratively without exposing raw client data, yet its behaviour under the highly heterogeneous, non-IID traffic distributions characteristic of real IoT deployments remains poorly characterised.

This repository presents a controlled empirical study that compares four training conditions — Centralized, Controlled IID FedAvg, Device-Level Non-IID FedAvg, and Local-Only — using a lightweight SmallMLP (9,537 parameters) on the N-BaIoT botnet detection dataset. All experiments are executed on CPU with a single random seed (42), a 3-round federated horizon, and 9 simulated IoT device clients spanning 7,062,606 observations across 115 features. Resource consumption (runtime and peak RSS) and communication overhead are measured systematically alongside predictive performance.

Key findings confirm that IID FedAvg achieves near-perfect detection (F1 = 0.9994), surpassing even the centralized baseline, while Device Non-IID FedAvg sustains strong but degraded performance (F1 = 0.9757) relative to IID conditions. Local-Only training achieves competitive per-device accuracy (F1 = 0.9714 macro) but cannot generalise across unseen device types. The full codebase, frozen data splits, persisted artefacts, publication figures, and a comprehensive reproducibility guide are included.

---

## Key Results

| Condition | Test Set | F1-Score | ROC-AUC |
|:---|:---|:---:|:---:|
| Centralized | Global | 0.997097 | 0.996899 |
| Controlled IID FedAvg (Round 3) | Global | 0.999430 | 0.999859 |
| Device Non-IID FedAvg (Round 3) | Global | 0.975696 | 0.992387 |
| Local-Only (Macro Average) | Per-device | 0.971422 | 0.994417 |

All metrics are evaluated on a frozen holdout test set (15% of data, 1,059,388 samples) using a fixed classification threshold of 0.5. The frozen 70/15/15 split was established with seed 42 prior to any training.

---

## Repository Structure

```
federated-iot-ids/
├── README.md
├── LICENSE                          # MIT License
├── CITATION.cff                     # Machine-readable citation metadata
├── requirements.txt
├── configs/
│   └── experiment.yaml              # Centralised experiment configuration
├── src/
│   ├── data/                        # Dataset loading and split utilities
│   ├── evaluation/                  # Metric computation and evaluation routines
│   ├── fl/                          # FedAvg aggregation and client logic
│   ├── models/                      # SmallMLP architecture definition
│   └── utils/                       # Logging, seeding, resource monitoring
├── experiments/scripts/             # Entry-point scripts for all 4 conditions
├── tests/                           # 105 pytest unit and integration tests
├── data/processed/splits/           # Frozen split specifications (no raw data)
├── results/
│   ├── raw/                         # JSON execution logs for all runs
│   └── processed/publication/
│       ├── figures/                 # 6 publication-quality PNG figures
│       └── tables/                  # 6 CSV result tables
├── manuscript/
│   ├── manuscript.md                # Full research manuscript (Markdown)
│   ├── tables/                      # Manuscript-facing CSV tables
│   └── latex/                       # Self-contained LaTeX package for Overleaf
│       ├── main.tex
│       ├── references.bib
│       └── figures/                 # Local copies of all 6 figures
└── docs/
    ├── methodology/                 # Design decisions and architecture notes
    ├── experiments/                 # Reproducibility guide, release notes
    └── dataset/                     # Dataset boundary and provenance notes
```

---

## Installation

```bash
git clone https://github.com/Moazzam9/federated-iot-ids.git
cd federated-iot-ids

python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

**Python 3.12 is required.** All experiments were executed on Windows 10 Pro 64-bit with CPU-only PyTorch.

---

## Dataset

This repository does **not** include raw data. The N-BaIoT dataset must be downloaded separately from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/442/detection+of+iot+botnet+attacks+n+baiot).

Place the extracted device directories under `data/raw/` according to the structure documented in `docs/dataset/`. The frozen split index specifications in `data/processed/splits/` guarantee that all experiments use an identical 70/15/15 partition.

| Statistic | Value |
|:---|:---:|
| Total observations | 7,062,606 |
| Features per sample | 115 |
| IoT device types | 9 |
| Train / Validation / Test | 70% / 15% / 15% |
| Random seed | 42 |

---

## Reproducing Experiments

All four experimental conditions are reproduced by running the corresponding entry-point scripts. See `docs/experiments/reproducibility.md` for the complete step-by-step guide.

```bash
# 1. Prepare frozen data splits and fit the StandardScaler
python experiments/scripts/prepare_data.py

# 2. Centralized training
python experiments/scripts/run_centralized.py

# 3. Local-Only training (one model per device)
python experiments/scripts/run_local_only.py

# 4. Controlled IID FedAvg (3 rounds)
python experiments/scripts/run_fedavg_iid.py

# 5. Device Non-IID FedAvg (3 rounds)
python experiments/scripts/run_fedavg_noniid.py

# 6. Generate publication figures and tables
python experiments/scripts/run_analysis.py
```

Execution times are substantial due to dataset size (see Limitations). Results are written to `results/raw/` as JSON logs and processed into `results/processed/publication/`.

---

## Testing

```bash
# Windows (required flag avoids PermissionError on default temp directory)
.\.venv\Scripts\python.exe -m pytest --basetemp=build/pytest_tmp -q
```

**Expected result:** 105 passed.

---

## Experimental Scope and Limitations

- **Simulated federation:** All 9 IoT device clients are simulated within a single process on one machine. No real network communication occurs.
- **Pre-fitted scaler:** The `StandardScaler` is fitted centrally on training data. A federated or local normalisation strategy is not evaluated.
- **Communication accounting:** Communication overhead is computed analytically from model state-dict size (9,537 float32 parameters × 4 bytes). No actual socket traffic is measured.
- **Resource measurement scope:** Runtime and peak RSS are measured at the process level. GPU, network I/O, and physical hardware power consumption are not measured.
- **No privacy evaluation:** Differential privacy, secure aggregation, and adversarial robustness are outside the scope of this study.
- **Single seed, 3 rounds:** All results reflect one random seed (42) and a 3-round federated horizon. Statistical variance across seeds and long-run convergence are not characterised.

---

## Citation

If you use this software or reproducibility package, please cite it as:

```bibtex
@software{azam_federated_iot_ids_2026,
  author = {Azam, Moazzam},
  title  = {Federated Learning for Resource-Aware Intrusion Detection in IoT Networks},
  year   = {2026},
  doi    = {10.5281/zenodo.22941258},
  url    = {https://doi.org/10.5281/zenodo.22941258}
}
```

A `CITATION.cff` file is also provided for automated citation tool support.

---

## License

This project is licensed under the [MIT License](LICENSE).

---

## Acknowledgements

The N-BaIoT dataset was collected and made publicly available by Meidan et al. (2018): *N-BaIoT — Network-based detection of IoT botnet attacks using deep autoencoders*, IEEE Pervasive Computing, vol. 17, no. 3, pp. 12–22. The authors are gratefully acknowledged.
