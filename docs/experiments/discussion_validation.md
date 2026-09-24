# Discussion Section Validation Report

## Overview
This report documents the validation audit of **Section 6: Discussion** in `manuscript/manuscript.md` against the 15 scientific and methodological validation requirements specified for Stage 4 of the `federated-iot-ids` project.

---

## 1. Summary of Validation Checks

| Check ID | Validation Requirement | Target Condition / Rule | Verification Finding | Status |
| :---: | :--- | :--- | :--- | :---: |
| **VAL-01** | Numerical Value Precision | Match verified result artifacts exactly | All 18 referenced numerical metrics (F1, ROC-AUC, Loss, RSS MiB, Wall Time, Bytes) match raw JSON and manuscript tables with 100% accuracy. | **PASS** |
| **VAL-02** | Cautious Causal Phrasing | No unproven causal assertions | Phrased as "consistent with the effect of client-level data heterogeneity". Unproven causal claims completely avoided. | **PASS** |
| **VAL-03** | Absence of Privacy Claims | No formal privacy protection claims | Section 6.7 explicitly disclaims formal privacy guarantees for vanilla FedAvg without DP/Secure Aggregation. | **PASS** |
| **VAL-04** | Absence of Energy Claims | No battery or physical energy claims | Section 6.5 & 6.7 explicitly note RSS memory and wall time represent host process CPU execution, not physical energy consumption. | **PASS** |
| **VAL-05** | Logical Client Interpretation | No physical IoT hardware deployment claims | Section 6.7 explicitly clarifies that 9 clients represent simulated logical participants executing in a single-host environment. | **PASS** |
| **VAL-06** | Absence of Statistical Significance Claims | No p-values or significance claims | 0 statistical significance or confidence interval assertions made. | **PASS** |
| **VAL-07** | Single Random Seed Limitation | No multi-seed generalization claims | Section 6.7 explicitly states results reflect a single empirical run (seed 42); cross-seed variance was not measured. | **PASS** |
| **VAL-08** | Asymptotic Convergence Claims | No proof of full convergence from 3 rounds | Section 6.3 explicitly describes 3 rounds as an initial execution window rather than proof of complete asymptotic convergence. | **PASS** |
| **VAL-09** | Real-World IID Claims | No claim that IID represents real IoT | Section 6.1 & 6.2 describe Controlled IID as an artificial experimental control condition, not a real-world IoT deployment. | **PASS** |
| **VAL-10** | IID Runtime Overhead Explanation | No unprofiled causal assertion for runtime | Section 6.5 cites implementation notes on index-slicing data loading while explicitly stating formal profiling was not performed. | **PASS** |
| **VAL-11** | Test Scope Distinction | Distinguish global test set from device test subsets | Section 6.1 & 6.7 explicitly document that global models used the global test set, whereas local-only models used device test subsets. | **PASS** |
| **VAL-12** | Global Scaler Pre-fitting Scope | Document global StandardScaler pre-fitting | Section 6.7 explicitly documents global training pre-fitting as an experimental consistency choice, noting it is not fully decentralized. | **PASS** |
| **VAL-13** | Communication Accounting Scope | Distinguish payload model from network traffic | Section 6.5 & 6.7 explicitly clarify that payload figures represent theoretical model tensor accounting, not measured socket traffic. | **PASS** |
| **VAL-14** | Process RSS Memory Scope | Distinguish process RSS from edge device RAM | Section 6.5 & 6.7 explicitly clarify that RSS measures host CPU RAM allocated to the Python process tree, not edge device RAM limits. | **PASS** |
| **VAL-15** | Absence of Subjective Ranking Words | Exclude subjective words (`best`, `worst`, etc.) | 0 occurrences of `best`, `worst`, `superior`, `inferior`, `optimal`, `impressive`, `remarkable` in Section 6. | **PASS** |

---

## 2. Detailed Verification Results

### Numerical Metric Verification
* **Centralized Baseline Test F1 / ROC-AUC:** 0.997097 / 0.996899 (Verified against `final_test_evaluation.json`).
* **Controlled IID FedAvg Test F1 / ROC-AUC:** 0.999430 / 0.999859 (Verified against `final_test_evaluation.json`).
* **Device Non-IID FedAvg Test F1 / ROC-AUC:** 0.975696 / 0.992387 (Verified against `final_test_evaluation.json`).
* **Local-Only Macro / Weighted Test F1:** 0.971422 / 0.969684 (Verified against `final_test_evaluation.json`).
* **Local-Only Min / Max Test F1:** 0.913371 (Philips Baby Monitor) / 0.999599 (SimpleHome 1003 Security Camera) (Verified against `local_only_results.json`).
* **Communication Payload Bytes:** 38,148 B per state dict; 76,296 B per client/round; 686,664 B per 9-client round; 2,059,992 B 3-round total (~1.96 MiB / 2.06 MB) (Verified against `communication_measurement.json`).
* **Computational Runtimes:** 5,018.94 s (Centralized), 3,402.99 s (Local-Only), 4,034.84 s (Device Non-IID FedAvg), 16,290.68 s (Controlled IID FedAvg) (Verified against resource JSON files).
* **Peak Memory RSS:** 610.94 MiB (Centralized), 610.95 MiB (Local-Only), 621.54 MiB (Device Non-IID FedAvg), 668.36 MiB (Controlled IID FedAvg) (Verified against resource JSON files).

---

## 3. Overall Validation Status

**Final Validation Status: PASS**

All 15 validation points passed cleanly. Section 6 provides an analytical, objective interpretation of the verified experimental results without overclaims or unverified assumptions.
