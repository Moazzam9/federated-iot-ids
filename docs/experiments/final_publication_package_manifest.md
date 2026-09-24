# Final Publication Package Manifest — Stage 12

This document provides the official manifest and quality-control verification log for the final publication package of the research study:

> **Title:** *Federated Learning for Resource-Aware Intrusion Detection in IoT Networks: An Empirical Study of IID and Device-Level Non-IID Data*  
> **Repository:** `federated-iot-ids` (`https://github.com/Moazzam9/federated-iot-ids.git`)

---

## 1. Package Contents & Locations

### Primary Manuscript
- **File:** `manuscript/manuscript.md`
- **Word / Line Count:** ~630 lines, complete single-file manuscript.
- **Structure:** Title, Abstract, Keywords, 1. Introduction, 2. Research Questions, 3. Related Work & Background, 4. Experimental Setup, 5. Results, 6. Discussion, 7. Limitations & Threats to Validity, 8. Conclusion & Future Work, References (`[1]`–`[15]`).

### Publication Figures (`results/processed/publication/figures/`)
1. `fedavg_f1_convergence.png` — Validation F1-Score progression across rounds (IID vs Non-IID FedAvg).
2. `fedavg_roc_auc_convergence.png` — Validation ROC-AUC progression across rounds (IID vs Non-IID FedAvg).
3. `local_only_f1_by_device.png` — Local-Only test F1-Score breakdown across 9 individual IoT device clients.
4. `communication_payload.png` — Cumulative model parameter payload transmission volume across communication rounds.
5. `experiment_runtime.png` — Total CPU wall-clock execution runtimes across all four experimental conditions.
6. `peak_memory_usage.png` — Peak process-tree Resident Set Size (RSS memory in MiB) across experimental conditions.

### Publication Tables (`results/processed/publication/tables/` & Embedded Markdown)
1. `table_1_final_test_results` (`main_performance.csv`) — Global holdout test set results across Centralized, IID FedAvg, Non-IID FedAvg, and Local-Only baselines.
2. `table_2_fedavg_convergence` (`fedavg_convergence.csv`) — Round-by-round global validation metrics across 3 communication rounds.
3. `table_3_local_only_by_device` (`local_only_by_device.csv`) — Per-device test evaluation for 9 isolated Local-Only models.
4. `table_4_communication_accounting` (`communication_cost.csv`) — Model parameter state dict exchange volume accounting.
5. `table_5_resource_usage` (`resource_usage.csv`) — Process-tree RSS memory and CPU wall-clock execution runtimes.
6. `table_6_validation_vs_test` (`validation_to_test.csv`) — Numerical agreement between final validation and global holdout test metrics.

### Raw Execution JSON Artifacts (`results/raw/`)
- `centralized/centralized_training_history.json` — 3-epoch centralized training log.
- `local_only/local_only_results.json` — 9-device local-only training & test evaluation log.
- `device_non_iid_3round_seed42.json` — 3-round device-level Non-IID FedAvg training log.
- `iid_3round_seed42.json` — 3-round controlled IID FedAvg training log.
- `final_test_evaluation/final_test_evaluation.json` — Final holdout evaluation log across all models.
- `communication/communication_measurement.json` — Model payload state dict accounting log.
- `resource_measurement/*.json` — `psutil` process-tree RSS and wall-clock execution logs.

### Key Documentation & Verification Reports (`docs/`)
- `docs/experiments/reproducibility.md` — Primary step-by-step reproducibility guide.
- `docs/experiments/scientific_consistency_audit.md` — 29-area scientific consistency audit report.
- `docs/experiments/literature_audit.md` — Primary literature citation audit (`[1]`–`[15]`).
- `docs/methodology/experimental_setup.md` — Detailed technical setup and framework documentation.
- `README.md` — Top-level repository overview and quick-start entry point.

---

## 2. Headline Results & Verified Metrics Summary

All metrics reported in `manuscript/manuscript.md` have been cross-checked against raw JSON logs:

| Condition | Scope | F1-Score | ROC-AUC | Accuracy | Precision | Recall | Loss |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Centralized Baseline** | Global Test Set | **0.997097** | 0.996899 | 0.994635 | 0.994216 | 0.999994 | 0.051443 |
| **Controlled IID FedAvg (Rd 3)** | Global Test Set | **0.999430** | 0.999859 | 0.998949 | 0.999193 | 0.999667 | 0.005075 |
| **Device Non-IID FedAvg (Rd 3)** | Global Test Set | **0.975696** | 0.992387 | 0.954104 | 0.952555 | 0.999990 | 0.129260 |
| **Local-Only (Macro Avg)** | Device Subsets | **0.971422** | 0.994417 | 0.946415 | 0.945825 | 0.999971 | 0.313455 |
| **Local-Only (Weighted Avg)** | Device Subsets | **0.969684** | 0.994566 | 0.943251 | 0.942662 | 0.999976 | 0.315096 |

- **State Dict Payload:** 38,148 bytes per model transfer (9,537 float32 parameters $\times$ 4 bytes).
- **3-Round Cumulative Payload:** 2,059,992 bytes (~1.96 MiB / 2.06 MB) across 9 clients.
- **Host Resource Ranges:** Peak process-tree RSS = 610.94 MiB to 668.36 MiB; execution wall time = 3,402.99 s to 16,290.68 s on CPU.

---

## 3. Quality Control & Quality Assurance Results

- **Automated Test Suite:** `105 passed` (via `pytest --basetemp=build/pytest_tmp`).
- **Placeholder Inspection:** `0` unfinished placeholders found (`TODO`, `FIXME`, `TBD`, `PLACEHOLDER`, `XXX`, `DEBUG`).
- **Path Portability:** `0` unneeded personal machine paths found in publication-facing documents.
- **Numerical Precision:** `0` rounding or transcription discrepancies between manuscript text, CSV tables, and raw JSONs.
- **Internal Reference Scrub:** `0` internal reference phrases remaining.

---

## 4. Methodological Scope & Interpretation Boundaries

1. **Simulated Clients:** 9 logical participants running in single-host CPU simulation, not physical hardware edge devices.
2. **Pre-fitted Scaler:** `StandardScaler` fitted globally on pooled training data prior to federated partitioning.
3. **Model Payload Accounting:** Parameter exchange volume calculated from state dict size, not physical network socket traffic or transport protocol headers.
4. **Host RSS Resource Scope:** Memory and wall-clock times reflect Python CPU process execution on host workstation; no energy, power, or battery consumption measured.
5. **Absence of Privacy Guarantees:** Vanilla FedAvg avoids raw data centralization but does not evaluate Differential Privacy, Secure Aggregation, or resilience against poisoning/inference attacks.
6. **Execution Horizon:** Single random seed (42) across 3 federated rounds; long-term asymptotic convergence tails remain unmeasured.

---

## 5. Verification Status

The publication package assembly and final quality control for Stage 12 are **COMPLETE**. The repository is fully self-contained, structurally sound, reproducible, and ready for manuscript submission or internal archiving.
