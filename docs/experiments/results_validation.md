# Results Section Validation Report

## Overview
This report documents the empirical audit and numerical validation of **Section 5: Results** in `manuscript/manuscript.md` against raw and processed artifact JSON/CSV files in the `federated-iot-ids` repository.

---

## 1. Files & Artifacts Inspected

1. `results/raw/centralized/centralized_training_history.json`
2. `results/raw/local_only/local_only_results.json`
3. `results/raw/device_non_iid_3round_seed42.json`
4. `results/raw/iid_3round_seed42.json`
5. `results/raw/final_test_evaluation/final_test_evaluation.json`
6. `results/raw/communication/communication_measurement.json`
7. `results/raw/resource_measurement/centralized_3epoch.json`
8. `results/raw/resource_measurement/local_only_3epoch.json`
9. `results/raw/resource_measurement/device_non_iid_3round_seed42.json`
10. `results/raw/resource_measurement/iid_3round_seed42.json`
11. `results/processed/publication/publication_results.json`
12. `results/processed/publication/tables/main_performance.csv`
13. `results/processed/publication/tables/fedavg_convergence.csv`
14. `results/processed/publication/tables/local_only_by_device.csv`
15. `results/processed/publication/tables/communication_cost.csv`
16. `results/processed/publication/tables/resource_usage.csv`
17. `results/processed/publication/tables/validation_to_test.csv`
18. `manuscript/tables/*.csv` (Copied manuscript table files)

---

## 2. Category-by-Category Numerical Checks

| Result Category | Target Table / Subsection | Source Artifact Files | Checks Performed | Status |
| :--- | :--- | :--- | :---: | :---: |
| **Overall Final-Test Performance** | Section 5.1 & Table 1 | `final_test_evaluation.json`, `main_performance.csv` | 30 | **PASS** |
| **FedAvg Convergence** | Section 5.2 & Table 2 | `device_non_iid_3round_seed42.json`, `iid_3round_seed42.json`, `fedavg_convergence.csv` | 36 | **PASS** |
| **Local-Only Device-Level Results** | Section 5.3 & Table 3 | `local_only_results.json`, `final_test_evaluation.json`, `local_only_by_device.csv` | 66 | **PASS** |
| **Communication Cost** | Section 5.4 & Table 4 | `communication_measurement.json`, `communication_cost.csv` | 20 | **PASS** |
| **Computational Resource Usage** | Section 5.5 & Table 5 | Resource measurement JSON files, `resource_usage.csv` | 20 | **PASS** |
| **Validation-to-Test Consistency** | Section 5.6 & Table 6 | `validation_to_test_comparison.csv` | 72 | **PASS** |
| **Total Numerical Checks** | All Section 5 Subsections | All Raw & Processed JSON/CSV Artifacts | **244** | **PASS** |

---

## 3. Detailed Check Verification Highlights

* **Final Test Set Evaluation (Table 1):**
  * Centralized: Loss 0.051443, Acc 0.994635, Prec 0.994216, Rec 0.999994, F1 0.997097, ROC-AUC 0.996899 (Verified against `final_test_evaluation.json` line 22-27).
  * Controlled IID FedAvg (Round 3): Loss 0.005075, Acc 0.998949, Prec 0.999193, Rec 0.999667, F1 0.999430, ROC-AUC 0.999859 (Verified against line 48-53).
  * Device Non-IID FedAvg (Round 3): Loss 0.129260, Acc 0.954104, Prec 0.952555, Rec 0.999990, F1 0.975696, ROC-AUC 0.992387 (Verified against line 35-40).
  * Local-Only Macro Average: Loss 0.313455, Acc 0.946415, Prec 0.945825, Rec 0.999971, F1 0.971422, ROC-AUC 0.994417 (Verified against line 162-167).
  * Local-Only Weighted Average: Loss 0.315096, Acc 0.943251, Prec 0.942662, Rec 0.999976, F1 0.969684, ROC-AUC 0.994566 (Verified against line 170-175).

* **FedAvg Validation Convergence (Table 2):**
  * Device Non-IID F1: Round 1 (0.959030) $\rightarrow$ Round 2 (0.962697) $\rightarrow$ Round 3 (0.975650), gain +0.016620.
  * Device Non-IID ROC-AUC: Round 1 (0.537014) $\rightarrow$ Round 2 (0.982908) $\rightarrow$ Round 3 (0.991824), gain +0.454809.
  * Controlled IID F1: Round 1 (0.998890) $\rightarrow$ Round 2 (0.999122) $\rightarrow$ Round 3 (0.999403), change +0.000513.

* **Communication Payload Accounting (Table 4):**
  * State dict size: 38,148 bytes (float32 $\times$ 9,537 parameters).
  * Per client per round: 76,296 bytes (Download + Upload).
  * All clients per round (9 clients): 686,664 bytes (~0.654854 MiB).
  * 3-Round experiment total: 2,059,992 bytes (~1.964561 MiB / 2.059992 MB).

* **Resource Runtimes & RAM RSS Memory (Table 5):**
  * Centralized (3 epochs): Peak RSS = 610.94 MiB, Wall Time = 5,018.94 s.
  * Local-Only (3 epochs): Peak RSS = 610.95 MiB, Wall Time = 3,402.99 s.
  * Device Non-IID FedAvg (3 rounds): Peak RSS = 621.54 MiB, Wall Time = 4,034.84 s.
  * Controlled IID FedAvg (3 rounds): Peak RSS = 668.36 MiB, Wall Time = 16,290.68 s.

---

## 4. Discrepancy Audit & Safeguards Verified

1. **Evaluation Scope Heterogeneity:** Explicitly documented in Section 5.1 and Table 1 note that Centralized and FedAvg global models are evaluated on the global test set (1,059,388 rows), whereas Local-Only models are evaluated on same-device test subsets.
2. **Communication Accounting vs Actual Network Traffic:** Explicitly disclaimed in Section 5.4 that communication byte figures represent state dictionary payload calculations and not actual network socket traffic or transport protocol overhead.
3. **Resource RSS vs Physical Energy:** Explicitly disclaimed in Section 5.5 that memory metrics reflect host process tree RSS on CPU execution and not edge device RAM limits or energy consumption.
4. **Subjective Ranking & Overclaims:** Verified that subjective ranking words ("best", "worst", "superior", "inferior", "optimal") and unsupported privacy/significance claims were completely avoided.

---

## 5. Final Validation Status

**Overall Status: PASS**

All 244 numerical verification checks passed. No discrepancies or unverified values exist between `manuscript/manuscript.md` and the underlying experiment artifacts.
