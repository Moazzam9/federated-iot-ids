# Limitations and Threats to Validity — Validation Documentation

This document records the systematic verification of Section 7 ("Limitations and Threats to Validity") of the manuscript against 22 explicit validation requirements for the study.

---

## Validation Checklist

| Item | Requirement / Scope | Status | Manuscript Section | Verification Details |
| :---: | :--- | :---: | :---: | :--- |
| **1** | Single Seed (42) & 3-Round Horizon | **PASS** | Section 7.1 | Explicitly states all primary experiments used `seed = 42` and 3 rounds; clarifies that 10-round figures in Section 5.4 are linear payload projections, not executed 10-round runs. |
| **2** | Global StandardScaler Pre-fitting | **PASS** | Section 7.2 | Clarifies that `StandardScaler` ($\mu, \sigma$) was pre-fitted globally on pooled training data (4,943,824 rows) to prevent scaling disparities from confounding baselines, and notes this requires prior centralized access. |
| **3** | Simulated Client Environment | **PASS** | Section 7.3 | Explicitly defines the 9 logical clients as host process simulations rather than physical edge hardware nodes. |
| **4** | Evaluation Scope Differences | **PASS** | Section 7.4 | Details that global models (Centralized, FedAvg) were evaluated on the 1.06M global test set, whereas Local-Only models were evaluated on same-device test subsets. |
| **5** | Communication Accounting Model Boundaries | **PASS** | Section 7.5 | Clarifies that communication metrics represent theoretical state dictionary payload calculations (`state_dict` size: 38,148 bytes) rather than physical socket traffic or network packet captures. |
| **6** | Computational and Hardware Scope | **PASS** | Section 7.6 | Notes that process RSS memory (610.94–668.36 MiB) and execution runtimes reflect host CPU workstation execution, not edge hardware RAM or physical energy/battery metrics. |
| **7** | Dataset Characteristics & Duplicate Audit | **PASS** | Section 7.7 | Documents N-BaIoT dataset properties (7,062,606 rows, 2,482,676 unique, 4,579,930 duplicates) and confirms duplicate-aware grouping by `(device, feature_hash)` across train/val/test splits. |
| **8** | Model & Algorithm Scope | **PASS** | Section 7.8 | Identifies specific scope bounds: `SmallMLP` (9,537 parameters), Adam ($\alpha=0.001$, batch size 256), binary classification, and standard FedAvg without evaluating CNNs, LSTMs, trees, or non-IID algorithms (FedProx, SCAFFOLD). |
| **9** | Privacy & Security Scope | **PASS** | Section 7.9 | Explicitly disclaims formal privacy guarantees: vanilla FedAvg without Differential Privacy ($\epsilon, \delta$), Secure Aggregation, or empirical privacy/poisoning attack testing. |
| **10** | Controlled IID Condition | **PASS** | Section 7.10 | Disclaims Controlled IID FedAvg as an artificial experimental control baseline rather than a realistic IoT traffic model. |
| **11** | Full Client Participation | **PASS** | Section 7.11 | Notes that full client participation ($C=1.0$) was assumed without modeling client dropout, stragglers, or intermittent connectivity. |
| **12** | Physical Deployment Disclaimers | **PASS** | Section 7.3, 7.6 | Disclaims physical micro-controllers, wireless channels (Wi-Fi, Ethernet, Cellular), network middleboxes, real-time packet capture, edge gateway topologies, and battery constraints. |
| **13** | No Overclaiming of Statistical Significance | **PASS** | Section 7.1 | Confirms no claims of statistical significance or confidence intervals across multiple random seeds were made. |
| **14** | No Overclaiming of Formal Privacy | **PASS** | Section 7.9 | Confirms no claims of Differential Privacy, cryptographic protection, or security attack resistance were made. |
| **15** | No Overclaiming of Hardware Energy/RAM | **PASS** | Section 7.6 | Confirms host CPU process RSS memory and runtime are not claimed to represent physical edge device RAM or battery consumption. |
| **16** | Objective Language & No Subjective Ranking | **PASS** | Sections 7.1–7.11 | Verified that subjective ranking words (`best`, `worst`, `superior`, `inferior`, `optimal`) are absent from evaluation commentary. |
| **17** | Manuscript & Repository Consistency | **PASS** | Sections 7.1–7.11 | Verified alignment with `docs/experiments/experimental_setup.md` and repository implementation documentation (`data/processed/splits/split_specification.txt`). |
| **18** | Numerical Precision & Exact Alignment | **PASS** | Sections 7.1–7.11 | All referenced numbers (7,062,606 rows; 4,943,824 train; 1,059,388 test; 9,537 params; 38,148 bytes payload; 610.94–668.36 MiB RSS) match Stage 2, 3, and 4 empirical records exactly. |
| **19** | Validation vs Test Metric Clarity | **PASS** | Section 7.1, 7.4 | Validation trajectories and holdout test set evaluations are explicitly distinguished. |
| **20** | Local-Only vs Global Evaluation Clarity | **PASS** | Section 7.4 | Local-only same-device test evaluations and global model holdout test evaluations are clearly separated in scope. |
| **21** | Internal Threats to Validity | **PASS** | Section 7.11 | Clearly documents internal validity threats: single seed, pre-fitted global scaler, data loader indexing runtime overhead. |
| **22** | External Threats to Validity | **PASS** | Section 7.11 | Clearly documents external validity threats: N-BaIoT dataset scope, host CPU simulation environment, SmallMLP architecture, standard FedAvg algorithm. |

---

## Conclusion

All 22 validation items for Section 7 ("Limitations and Threats to Validity") have been verified. The manuscript maintains scientific accuracy, avoids overclaiming, accurately presents experimental boundaries, and aligns with empirical repository artifacts.
