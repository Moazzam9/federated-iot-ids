# Final Repository & Archival Preparation Summary — Stage 14

This document provides the official preparation summary and quality-control verification log for the final repository and archive pass of the research project:

> **Project Title:** *Federated Learning for Resource-Aware Intrusion Detection in IoT Networks: An Empirical Study of IID and Device-Level Non-IID Data*  
> **Repository:** `federated-iot-ids` (`https://github.com/Moazzam9/federated-iot-ids.git`)  
> **Branch:** `main`  
> **Prepared Version:** `v1.0.0` (unreleased/archival-ready)

---

## 1. Repository Identity & Metadata

- **Full Project Title:** *Federated Learning for Resource-Aware Intrusion Detection in IoT Networks: An Empirical Study of IID and Device-Level Non-IID Data* (verified uniform across `README.md`, `manuscript/manuscript.md`, and `CITATION.cff`).
- **Copyright & Author:** Copyright (c) 2026 Moazzam Azam (`Moazzam9`). No fabricated ORCID, DOI, university affiliation, funding source, or journal/conference names.
- **Publication Status:** Pre-print / research artifact package under preparation. Not yet peer-reviewed or published in a journal/conference. No DOI assigned (`DOI: NOT YET CREATED`).
- **License:** MIT License (`LICENSE`) covering project code and reproducibility package.
- **External Dataset License Notice:** The N-BaIoT dataset is distributed separately under its original terms (UCI ML Repository). It is not covered by or bundled inside this repository's MIT License.

---

## 2. Dataset Boundary & Privacy Audit

- **Raw Dataset Exclusion:** Raw N-BaIoT CSV files and ZIP archives are **not** tracked by Git (`git ls-files` verified).
- **Temporary Extraction Exclusion:** Local temp directories (e.g. `D:\federated-iot-temp`) are excluded from tracking.
- **Dataset Instructions:** Clear acquisition instructions are provided in `docs/experiments/reproducibility.md` pointing to the UCI Machine Learning Repository.
- **Secrets & Credentials Scan:** `0` API keys, tokens, passwords, or private keys found in tracked repository files.
- **Private Machine Paths:** `0` local machine paths (`D:\federated-iot-ids`, `C:\Users\`) present in publication-facing documents (`manuscript/`, `README.md`, `docs/`, `CITATION.cff`).

---

## 3. Verified Headline Empirical Metrics

All reported numbers are verified against raw JSON execution logs in `results/raw/`:

| Condition | Evaluation Scope | F1-Score | ROC-AUC | Accuracy | Precision | Recall | Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | Global Test Set (1,059,388) | **0.997097** | 0.996899 | 0.994635 | 0.994216 | 0.999994 | 0.051443 |
| **Controlled IID FedAvg (Rd 3)** | Global Test Set (1,059,388) | **0.999430** | 0.999859 | 0.998949 | 0.999193 | 0.999667 | 0.005075 |
| **Device Non-IID FedAvg (Rd 3)** | Global Test Set (1,059,388) | **0.975696** | 0.992387 | 0.954104 | 0.952555 | 0.999990 | 0.129260 |
| **Local-Only (Macro Avg)** | Same-Device Test Subsets | **0.971422** | 0.994417 | 0.946415 | 0.945825 | 0.999971 | 0.313455 |
| **Local-Only (Weighted Avg)** | Same-Device Test Subsets | **0.969684** | 0.994566 | 0.943251 | 0.942662 | 0.999976 | 0.315096 |

- **Model Parameters:** `SmallMLP` — 9,537 parameters (`115 → 64 → 32 → 1`).
- **Parameter Payload Accounting:** 38,148 bytes per model transfer; 2,059,992 bytes (~1.96 MiB / 2.06 MB) cumulative over 3 rounds across 9 clients.
- **Resource Measurement Scope:** Peak process-tree RSS memory (610.94 MiB to 668.36 MiB) and host CPU execution wall-clock time (3,402.99 s to 16,290.68 s).

---

## 4. Explicit Scope & Claim Boundaries

1. **Simulated Logical Participants:** 9 clients executed as simulated logical processes on a single host CPU (Intel Core i7-6820HQ), not physical embedded edge hardware.
2. **Global Training Preprocessor:** `StandardScaler` fitted globally on pooled training data prior to federated partitioning.
3. **Parameter Payload Accounting:** Exchange volumes are calculated from model state dict sizes, not physical network socket traffic or transport protocol headers.
4. **Host CPU Resource Usage:** Resource figures reflect host-process RSS memory and wall time; no energy, power, or battery consumption was measured.
5. **Vanilla FedAvg Limits:** Evaluates vanilla FedAvg without formal Differential Privacy, Secure Aggregation, or attack-resilience mechanisms.
6. **Execution Horizon:** Based on single random seed (42) and a 3-round federated training horizon.

---

## 5. Artifact & Verification Status

- **Automated Test Suite:** `105 passed` via `pytest --basetemp=build/pytest_tmp -q`.
- **Publication Figures:** 6 PNG figures tracked in `figures/` and `results/processed/publication/figures/`.
- **Publication Tables:** 6 CSV tables tracked in `results/processed/publication/tables/`.
- **Reproducibility Documentation:** `docs/experiments/reproducibility.md` present and verified.
- **Manuscript:** `manuscript/manuscript.md` complete and verified (628 lines).
- **GitHub Release / Archival Metadata:** Release notes (`docs/experiments/github_release_notes.md`), release checklist (`docs/experiments/github_release_checklist.md`), and software citation (`CITATION.cff`) verified.

---

## 6. Archival Readiness & External Status

- **External Upload / Publication:** **NOT PERFORMED** (No Zenodo deposit, no external upload).
- **DOI Creation:** **NOT CREATED**.
- **GitHub Release:** **NOT CREATED** (Prepared for tag `v1.0.0` when authorized).
- **Git Tag:** **NOT CREATED**.
- **Stage 15 Status:** **NOT STARTED**.

---

## 7. Verification Conclusion

Stage 14 final repository and archive preparation is **COMPLETE**. The repository is clean, scientifically consistent, fully reproducible, correctly licensed under the MIT License, and suitable for long-term archival, future repository citation, or manuscript submission.
