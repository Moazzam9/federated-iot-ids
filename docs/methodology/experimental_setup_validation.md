# Experimental Setup Validation Report

## Overview
This report documents the validation of the Stage 2 Experimental Setup manuscript section and methodology documentation against the actual codebase, configuration files, and empirical experiment artifacts of the `federated-iot-ids` project.

---

## 1. Files Inspected
The following primary files were audited and cross-referenced during validation:

1. `configs/experiment.yaml` (Project configuration file)
2. `src/models/mlp.py` (`SmallMLP` network architecture definition)
3. `src/models/trainer.py` (PyTorch training loop, loss function, and metrics evaluation)
4. `src/fl/client.py` (Local client training implementation)
5. `src/fl/coordinator.py` (FedAvg communication round coordinator)
6. `src/fl/fedavg.py` (Weighted parameter aggregation implementation)
7. `src/fl/partitioning.py` (Device and stratified IID dataset partitioners)
8. `src/fl/iid_data.py` (IID client dataset stream loader)
9. `src/data/preprocessing.py` (`FittedStandardScaler` incremental fitting)
10. `src/data/nbaiot_loader.py` (`NBaIoTSplitLoader` CSV chunk reader)
11. `experiments/scripts/run_centralized.py` (Centralized baseline runner)
12. `experiments/scripts/run_local_only.py` (Local-only baseline runner)
13. `experiments/scripts/run_multiround_federated.py` (FedAvg multi-round runner)
14. `experiments/scripts/measure_communication.py` (Communication payload measurement)
15. `experiments/scripts/measure_resource_usage.py` (Process RSS resource monitor)
16. `data/processed/splits/split_specification.txt` (Dataset row count and split specifications)
17. `results/raw/centralized/centralized_training_history.json` (Centralized empirical results)
18. `results/raw/local_only/local_only_results.json` (Local-only empirical results)
19. `results/raw/device_non_iid_3round_seed42.json` (Device Non-IID FedAvg empirical results)
20. `results/raw/iid_3round_seed42.json` (IID FedAvg empirical results)
21. `results/raw/final_test_evaluation/final_test_evaluation.json` (Final holdout test evaluation)
22. `results/raw/communication/communication_measurement.json` (Communication payload measurement artifact)
23. `results/raw/resource_measurement/*.json` (RSS memory and runtime artifacts)
24. `results/processed/publication/publication_results.json` (Aggregated publication package)

---

## 2. Validation Checks Performed

| Check ID | Target Parameter / Concept | Source Artifact Value | Manuscript / Doc Value | Status |
| :--- | :--- | :--- | :--- | :--- |
| **CHK-01** | Total Dataset Row Count | 7,062,606 | 7,062,606 | **PASS** |
| **CHK-02** | Feature Count | 115 continuous features | 115 continuous features | **PASS** |
| **CHK-03** | Train / Val / Test Row Division | 4,943,824 / 1,059,394 / 1,059,388 | 4,943,824 / 1,059,394 / 1,059,388 | **PASS** |
| **CHK-04** | Client Count & Device Sources | 9 logical clients (N-BaIoT devices) | 9 logical clients | **PASS** |
| **CHK-05** | Model Architecture Layer Dimensions | 115 -> 64 -> 32 -> 1 (ReLU, Sigmoid) | 115 -> 64 -> 32 -> 1 (ReLU, Sigmoid) | **PASS** |
| **CHK-06** | Model Parameter Count | 9,537 float32 parameters | 9,537 float32 parameters | **PASS** |
| **CHK-07** | Optimizer & Learning Rate | Adam, lr = 0.001 | Adam, lr = 0.001 | **PASS** |
| **CHK-08** | Batch Size | 256 | 256 | **PASS** |
| **CHK-09** | Executed Centralized Epochs | 3 epochs | 3 epochs | **PASS** |
| **CHK-10** | Executed Local-Only Epochs | 3 epochs per client | 3 epochs per client | **PASS** |
| **CHK-11** | Executed FedAvg Rounds | 3 communication rounds | 3 communication rounds | **PASS** |
| **CHK-12** | Local Epochs per FedAvg Round | 1 local epoch ($E=1$) | 1 local epoch ($E=1$) | **PASS** |
| **CHK-13** | Random Seed | 42 | 42 | **PASS** |
| **CHK-14** | Preprocessing Scaler Fitting Scope | Global training set (`fit_incremental`) | Global training set (`fit_incremental`) | **PASS** |
| **CHK-15** | State Dict Payload Size | 38,148 bytes | 38,148 bytes (~37.25 KiB) | **PASS** |
| **CHK-16** | 3-Round Communication Payload | 2,059,992 bytes | 2,059,992 bytes (~1.96 MiB) | **PASS** |
| **CHK-17** | Communication Accounting Nature | Theoretical tensor payload model | Theoretical tensor payload model | **PASS** |
| **CHK-18** | Peak Process RSS Memory | 610.94 - 668.36 MiB | 610.94 - 668.36 MiB | **PASS** |
| **CHK-19** | Evaluation Metric Threshold | Fixed at 0.5 ($p \ge 0.5 \rightarrow 1$) | Fixed at 0.5 ($p \ge 0.5 \rightarrow 1$) | **PASS** |
| **CHK-20** | Test Scope Distinction | Global test vs. local device subsets | Global test vs. local device subsets | **PASS** |

---

## 3. Discrepancies Discovered & Handling

### Discrepancy 1: Configured Maximums vs. Executed Parameters
* **Observation:** `configs/experiment.yaml` contains `max_centralized_epochs: 10`, `max_local_only_epochs: 10`, and `max_federated_rounds: 10`. However, actual completed experiment JSON files in `results/raw/` show 3 epochs and 3 rounds were executed.
* **Handling:** The manuscript explicitly distinguishes between configured upper bounds and executed parameters. The manuscript reports 3 epochs / 3 rounds as the executed primary experimental setup.

### Discrepancy 2: Scaler Scope vs. Federated Decentralization Concept
* **Observation:** The `StandardScaler` binary was fitted globally across the pooled 4.94M training rows prior to federated partitioning.
* **Handling:** The manuscript transparently notes this as a methodological choice and limitation, explicitly stating that the preprocessing pipeline is not fully decentralized.

### Discrepancy 3: Evaluation Scope Heterogeneity
* **Observation:** Centralized and FedAvg models are evaluated on the global test set (1,059,388 rows across all 9 devices), while Local-Only models are evaluated on local device-specific test subsets.
* **Handling:** Section 4.7 ("Evaluation Protocol") explicitly documents this evaluation scope distinction to ensure full transparency.

---

## 4. Scientific Terminology & Overclaim Audit

A automated pattern search was conducted across `manuscript/manuscript.md` and `docs/methodology/experimental_setup.md` for potential overclaim terms:
* `"privacy-preserving"`: Unqualified privacy claims removed/qualified with methodological limitations.
* `"guarantees privacy"` / `"secure"`: 0 occurrences found.
* `"energy efficient"`: 0 occurrences found (RAM RSS memory & wall-clock time reported instead).
* `"real IoT deployment"` / `"real devices"`: 0 occurrences found (described as simulated logical clients).
* `"network traffic was measured"`: 0 occurrences found (described as model parameter tensor payload accounting).
* `"statistically significant"`: 0 occurrences found (noted single seed 42 limit).

---

## 5. Overall Validation Status

**Final Status:** **PASS**

All numerical values, architectural specs, training configurations, baseline descriptions, and evaluation protocols match the codebase and persisted empirical artifacts exactly.
