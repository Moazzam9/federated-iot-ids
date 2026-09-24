# Scientific Consistency Audit — Stage 9

This document records the comprehensive scientific consistency audit of Publication 02 (`manuscript/manuscript.md`) against the authoritative implementation codebase, execution artifacts, and methodology documentation.

---

## 1. Audit Scope & Primary Question

**Primary Audit Question:**
> *"Does every important scientific statement in the manuscript accurately describe what was actually implemented, measured, and demonstrated?"*

This audit evaluates the complete assembled manuscript (`manuscript/manuscript.md`) to verify that all empirical values, methodological descriptions, theoretical disclaimers, literature citations, and research statements strictly reflect actual codebase implementations and execution output logs.

---

## 2. Files and Artifacts Inspected

The following primary files and execution artifacts were systematically cross-inspected during this audit:

### Manuscript & Validation Documents
- `manuscript/manuscript.md` (Full assembled manuscript, 630+ lines)
- `docs/methodology/experimental_setup.md`
- `docs/methodology/federated_learning_framework.md`
- `docs/methodology/experimental_setup_validation.md`
- `docs/experiments/results_validation.md`
- `docs/experiments/discussion_validation.md`
- `docs/experiments/limitations_validation.md`
- `docs/experiments/conclusion_validation.md`
- `docs/experiments/literature_audit.md`
- `docs/experiments/manuscript_assembly_validation.md`

### Raw Execution JSON Artifacts
- `results/raw/centralized/centralized_training_history.json`
- `results/raw/local_only/local_only_results.json`
- `results/raw/device_non_iid_3round_seed42.json`
- `results/raw/iid_3round_seed42.json`
- `results/raw/final_test_evaluation/final_test_evaluation.json`
- `results/raw/communication/communication_measurement.json`
- `results/raw/resource_measurement/centralized_3epoch.json`
- `results/raw/resource_measurement/local_only_3epoch.json`
- `results/raw/resource_measurement/device_non_iid_3round_seed42.json`
- `results/raw/resource_measurement/iid_3round_seed42.json`

### Source Code Modules
- `src/data/nbaiot_loader.py`
- `src/data/preprocessing.py`
- `src/data/torch_data.py`
- `src/models/mlp.py`
- `src/models/trainer.py`
- `src/fl/partitioning.py`
- `src/fl/iid_assignment.py`
- `src/fl/fedavg.py`
- `src/fl/client.py`
- `src/fl/coordinator.py`

---

## 3. Authoritative Evidence Hierarchy

When evaluating claims, the following hierarchy of authority was strictly enforced:

1. **Level 1 (Highest Authority):** Raw Execution JSON Logs (`results/raw/`) and Core Source Code (`src/`).
2. **Level 2:** Dataset Split Specifications & Binary Preprocessing Files (`data/processed/`).
3. **Level 3:** Validation Summary Documents (`docs/`).
4. **Level 4:** Assembled Manuscript Text (`manuscript/manuscript.md`).

Any discrepancy between manuscript text and Level 1 raw execution JSON logs was marked as an issue requiring correction.

---

## 4. Dataset Consistency Audit

- **Total Dataset Observations:** 7,062,606 rows (Verified in `dataset_row_inventory.csv` & `split_specification.txt`).
- **Source CSV Files:** 89 raw CSV files across 9 commercial IoT device types (Verified).
- **Features:** 115 continuous numerical features (5 decay windows $\times$ 4 aggregation streams).
- **Binary Target:** Class `0` (Benign) vs Class `1` (Attack: Mirai & Gafgyt botnet vectors).
- **Client Count / Device Sources:** 9 simulated logical clients (Danmini Doorbell, Ecobee Thermostat, Ennio Doorbell, Philips Baby Monitor, Provision 737E Camera, Provision 838 Camera, Samsung Webcam, SimpleHome 1002 Camera, SimpleHome 1003 Camera).
- **Audit Result:** **PASS**. All dataset counts and device names in `manuscript.md` match the underlying dataset inventory exactly.

---

## 5. Split Consistency Audit

- **Split Percentages:** 70% Training / 15% Validation / 15% Test.
- **Row Allocations:**
  - Training Set: 4,943,824 rows (70.00%)
  - Validation Set: 1,059,394 rows (15.00%)
  - Test Set: 1,059,388 rows (15.00%)
  - Total: 7,062,606 rows (100.00%)
- **Split Grouping Unit:** `(device, feature_hash)` within each IoT device source (seed 42).
- **Leakage Prevention Scope:** Identical feature vectors within the same device partition are assigned indivisibly to one split (train, val, or test).
- **Audit Finding (Scope Note):** Manuscript Section 4.1 and Section 7.7 correctly note that duplicate grouping was performed *within each device partition*, rather than globally across devices. Line 59 in Section 1.6 should be slightly clarified in Stage 10 to ensure readers understand grouping was applied within each device partition.
- **Audit Result:** **PASS** (with minor clarification recommendation for Line 59).

---

## 6. Preprocessing Audit

- **Scaler Type:** `StandardScaler` (zero mean, unit variance).
- **Fitting Protocol:** Fitted incrementally (`fit_training_scaler_incremental`) over the 4,943,824 pooled training rows prior to federated client partitioning (`training_standard_scaler.pkl`).
- **Transformation:** Validation and test partitions transformed using pre-fitted training statistics ($\mu, \sigma$) without refitting.
- **Federated Limitation:** Because feature statistics were computed centrally prior to client partitioning, this does not represent a fully decentralized edge pipeline.
- **Audit Result:** **PASS**. The scaler implementation is accurately documented in Sections 4.2, 6.7, and 7.2.

---

## 7. Model Architecture Audit

- **Architecture:** `SmallMLP` (`src/models/mlp.py`).
- **Layers:** 115 input features $\rightarrow$ Linear(115, 64) $\rightarrow$ ReLU $\rightarrow$ Linear(64, 32) $\rightarrow$ ReLU $\rightarrow$ Linear(32, 1) $\rightarrow$ Sigmoid.
- **Parameter Count Calculation:**
  - Layer 1: $(115 \times 64) + 64 = 7,424$
  - Layer 2: $(64 \times 32) + 32 = 2,080$
  - Output Layer: $(32 \times 1) + 1 = 33$
  - Total Trainable Parameters: $7,424 + 2,080 + 33 = \mathbf{9,537}$.
- **Optimizer & Config:** Adam ($\alpha = 0.001$, batch size 256, loss BCE, seed 42, CPU execution).
- **Audit Result:** **PASS**. Exact agreement across `src/models/mlp.py`, configuration YAMLs, raw JSONs, and manuscript text.

---

## 8. Federated Algorithm Audit

- **Algorithm:** Vanilla Federated Averaging (FedAvg) (`weighted_fedavg` in `src/fl/fedavg.py`).
- **Aggregation:** Sample-count weighted parameter state dict averaging:
  $$W^r = \sum_{k=1}^K \frac{n_k}{N} W_k^r$$
- **Participation:** Full client participation ($C = 1.0$, 9 out of 9 clients participating in every round).
- **Executed Parameters:** 3 communication rounds, $E = 1$ local epoch per client per round.
- **Exchanged Data:** Model parameter state dictionaries (`state_dict`), not gradients.
- **Audit Result:** **PASS**. No unexecuted algorithms (FedProx, SCAFFOLD) are presented as executed.

---

## 9. IID Condition Audit

- **Implementation:** `make_stratified_iid_partitions` in `src/fl/partitioning.py`.
- **Partitioning Method:** Class-stratified redistribution of the 4,943,824 training observations across 9 simulated logical clients (~549,313 to 549,315 rows per client).
- **Nature of Partition:** An artificial experimental control condition to isolate non-IID distribution effects, not a naturally occurring IoT traffic distribution.
- **Terminology Check:** Manuscript correctly uses "Controlled IID" or "class-stratified IID partition".
- **Audit Result:** **PASS**.

---

## 10. Device-Level Non-IID Condition Audit

- **Implementation:** Direct 1-to-1 mapping of the 9 commercial IoT device sources in N-BaIoT to 9 simulated logical clients.
- **Client Sizes:** 248,850 rows (`Ennio_Doorbell`) to 769,074 rows (`Philips_B120N10_Baby_Monitor`).
- **Nature of Partition:** Models real-world device distribution skew arising from distinct device hardware types and traffic usage patterns.
- **Audit Result:** **PASS**.

---

## 11. Centralized Baseline Results Audit

- **Source JSON:** `results/raw/centralized/centralized_training_history.json` and `results/raw/final_test_evaluation/final_test_evaluation.json`.
- **Global Test Set Metrics (1,059,388 samples):**
  - Loss: `0.051443154082` $\rightarrow$ **0.051443**
  - Accuracy: `0.994634638112` $\rightarrow$ **0.994635**
  - Precision: `0.994215984785` $\rightarrow$ **0.994216**
  - Recall: `0.999993852453` $\rightarrow$ **0.999994**
  - F1-Score: `0.997096548448` $\rightarrow$ **0.997097**
  - ROC-AUC: `0.996898914772` $\rightarrow$ **0.996899**
- **Validation Epoch 3 Metrics (1,059,394 samples):**
  - Loss: `0.052471`, Accuracy: `0.994549`, Precision: `0.994122`, Recall: `0.999996`, F1: `0.997050`, ROC-AUC: `0.996878`.
- **Audit Result:** **PASS**. All manuscript numbers match raw execution JSON files exactly.

---

## 12. Local-Only Results Audit

- **Source JSON:** `results/raw/local_only/local_only_results.json` and `results/raw/final_test_evaluation/final_test_evaluation.json`.
- **Macro-Averaged Test Metrics:**
  - Loss: `0.313455`, Accuracy: `0.946415`, Precision: `0.945825`, Recall: `0.999971`, F1: `0.971422`, ROC-AUC: `0.994417`.
- **Sample-Weighted Test Metrics:**
  - Loss: `0.315096`, Accuracy: `0.943251`, Precision: `0.942662`, Recall: `0.999976`, F1: `0.969684`, ROC-AUC: `0.994566`.
- **Per-Device Test F1 Range:** `0.913371` (`Philips_Baby_Monitor`) to `0.999599` (`SimpleHome_1003`).
- **Evaluation Scope Distinction:** Evaluated on device-specific test subsets (53,325 to 164,801 rows per device), whereas Centralized and FedAvg were evaluated on the full global test set (1,059,388 rows).
- **Audit Result:** **PASS**. Evaluation scope distinction is explicitly stated in Sections 4.7, 5.1, 6.7, and 7.4.

---

## 13. Device-Level Non-IID FedAvg Results Audit

- **Source JSON:** `results/raw/device_non_iid_3round_seed42.json` and `results/raw/final_test_evaluation/final_test_evaluation.json`.
- **Validation Progress Across Rounds:**
  - Round 1: Loss = `0.828505`, F1 = `0.959030`, ROC-AUC = `0.537014`
  - Round 2: Loss = `0.283213`, F1 = `0.962697`, ROC-AUC = `0.982908`
  - Round 3: Loss = `0.133517`, F1 = `0.975650`, ROC-AUC = `0.991824`
- **Global Test Set (Round 3):**
  - Loss: `0.129260`, Accuracy: `0.954104`, Precision: `0.952555`, Recall: `0.999990`, F1: `0.975696`, ROC-AUC: `0.992387`.
- **Audit Result:** **PASS**. Values match raw JSONs. Round-to-round metric progression is correctly described as "improved over the three measured rounds".

---

## 14. Controlled IID FedAvg Results Audit

- **Source JSON:** `results/raw/iid_3round_seed42.json` and `results/raw/final_test_evaluation/final_test_evaluation.json`.
- **Validation Progress Across Rounds:**
  - Round 1: Loss = `0.009921`, F1 = `0.998890`, ROC-AUC = `0.999194`
  - Round 2: Loss = `0.006481`, F1 = `0.999122`, ROC-AUC = `0.999705`
  - Round 3: Loss = `0.005061`, F1 = `0.999403`, ROC-AUC = `0.999801`
- **Global Test Set (Round 3):**
  - Loss: `0.005075`, Accuracy: `0.998949`, Precision: `0.999193`, Recall: `0.999667`, F1: `0.999430`, ROC-AUC: `0.999859`.
- **Audit Result:** **PASS**. Values match raw JSONs. Validation vs test set distinction is strictly maintained.

---

## 15. Convergence-Language Audit

- **Inspection:** Searched `manuscript/manuscript.md` for `converged`, `convergence`, `asymptotic`, `optimization guarantee`.
- **Findings:**
  - The manuscript uses "convergence" in descriptive section headings ("FedAvg Convergence Across Rounds") and validation trajectory analyses.
  - Section 6.3 explicitly states: *"executing 3 communication rounds represents an initial execution window rather than proof of complete asymptotic convergence."*
  - Sections 7.1 and 8.1 reiterate that 3 rounds characterize early-stage metric progression rather than long-term asymptotic convergence.
- **Audit Result:** **PASS**. Convergence language is properly guarded.

---

## 16. Causal-Language Audit

- **Inspection:** Searched `manuscript/manuscript.md` for causal assertions (`causes`, `due to`, `leads to`, `drives`).
- **Findings:**
  - Most causal claims are appropriately phrased using cautious empirical language (*"consistent with"*, *"associated with"*, *"suggests"*).
  - *Minor Finding:* Line 422 in Section 6.3 states *"reflecting rapid early alignment due to homogeneous local data distributions."* The phrase "due to" expresses direct causation.
  - *Stage 10 Recommendation:* Soften "due to" in line 422 to *"consistent with homogeneous local data distributions."*
- **Audit Result:** **PASS** (with minor line 422 softening recommendation).

---

## 17. Privacy and Security Audit

- **Inspection:** Searched `manuscript/manuscript.md` for `privacy`, `secure`, `protection`, `Differential Privacy`, `Secure Aggregation`.
- **Findings:**
  - The manuscript explicitly disclaims formal privacy guarantees in the Abstract, Section 1.6, Section 4.10, Section 6.7, Section 7.9, Section 8.1, and Section 8.2.
  - Sections 7.9 and 8.2 explicitly state that standard FedAvg does not incorporate Differential Privacy or Secure Aggregation and does not evaluate resilience against gradient inversion or poisoning attacks.
- **Audit Result:** **PASS**. No false privacy or security claims exist.

---

## 18. Resource-Claim Audit

- **Inspection:** Searched `manuscript/manuscript.md` for resource claims (`resource-aware`, `efficient`, `lightweight`, `energy`, `battery`, `power`).
- **Findings:**
  - "Resource-aware" in the title is justified by measured model parameter payload accounting and host process-tree RSS memory monitoring.
  - Sections 4.9, 5.5, 6.7, 7.6, and 8.1 explicitly clarify that reported RSS memory and execution wall times reflect single-host CPU simulation and do NOT measure edge hardware RAM limits, micro-controller memory bounds, physical power draw, energy consumption, or battery drain.
- **Audit Result:** **PASS**. Resource claims are strictly bounded.

---

## 19. Communication Accounting Audit

- **State Dict Size:** 9,537 float32 parameters $\times$ 4 bytes = **38,148 bytes** (~37.25 KiB / 0.038148 MB).
- **Per Client / Round:** 1 download + 1 upload = **76,296 bytes** (~74.51 KiB).
- **Per Round (9 Clients):** $9 \times 76,296 = \mathbf{686,664 \text{ bytes}}$ (~0.655 MiB / 0.687 MB).
- **3-Round Experiment Total:** $3 \times 686,664 = \mathbf{2,059,992 \text{ bytes}}$ (~1.96 MiB / 2.06 MB).
- **10-Round Projection:** 6,866,640 bytes (~6.55 MiB / 6.87 MB) — explicitly marked as a linear projection.
- **Disclaimers:** Sections 4.8, 5.4, 6.7, 7.5 explicitly document that these figures represent theoretical model tensor payload accounting and do NOT represent measured network socket traffic, TCP/IP headers, TLS framing, gRPC, serialization, latency, or packet drops.
- **Audit Result:** **PASS**.

---

## 20. Resource Measurement Audit

- **Centralized Baseline (3 epochs):** Peak RSS = **610.94 MiB** (640,614,400 bytes), Wall Time = **5,018.94 s** (~83.65 min).
- **Local-Only Baseline (3 epochs):** Peak RSS = **610.95 MiB** (640,630,784 bytes), Wall Time = **3,402.99 s** (~56.72 min).
- **Device Non-IID FedAvg (3 rounds):** Peak RSS = **621.54 MiB** (651,730,944 bytes), Wall Time = **4,034.84 s** (~67.25 min).
- **Controlled IID FedAvg (3 rounds):** Peak RSS = **668.36 MiB** (700,821,504 bytes), Wall Time = **16,290.68 s** (~271.51 min).
- **Runtime Anomaly Explanation:** Controlled IID wall time (~4.53 hours) was higher than Non-IID (~1.12 hours). Sections 5.5 and 6.5 accurately attribute this to indexing overhead in the custom PyTorch IID data partitioner during local epoch iteration, while explicitly noting that a dedicated profiling study was not performed.
- **Audit Result:** **PASS**.

---

## 21. Duplicate Structure Audit

- **Audit Totals:** 7,062,606 total observations; 2,482,676 unique feature vectors; 4,579,930 duplicate rows across 1,891,636 duplicate groups; 0 cross-label conflicts; 1,871,053 cross-device duplicate vector groups.
- **Grouping Rule:** Enforced `(device, feature_hash)` grouping within each device partition (`split_specification.txt`).
- **Audit Result:** **PASS**. Manuscript distinguishes duplicate rows from unique feature vectors and documents split grouping details accurately.

---

## 22. Evaluation Scope Audit

- **Global Scope Models (Centralized, IID FedAvg, Device Non-IID FedAvg):** Evaluated on frozen global test set (1,059,388 samples across all 9 devices).
- **Local Scope Models (Local-Only):** Evaluated on device-specific test subsets (53,325 to 164,801 samples per device).
- **Scope Disclaimers:** Sections 4.7, 5.1, 6.7, and 7.4 explicitly note that Local-Only metrics evaluate local device specialization, while global models evaluate network-wide generalization.
- **Audit Result:** **PASS**.

---

## 23. Literature Claims Audit

- **Inventory:** 15 verified primary references (`[1]`–`[15]`).
- **Usage Mapping:**
  - `[1]` McMahan et al. (FedAvg algorithm formulation & baseline)
  - `[2]` Meidan et al. (N-BaIoT dataset collection)
  - `[3]` Zarpelão et al. & `[4]` Chaabouni et al. (IoT IDS surveys)
  - `[5]` Zhao et al., `[6]` Li et al., `[7]` Kairouz et al. (Non-IID FL challenges & surveys)
  - `[8]` Arp et al. (Security ML pitfalls & data leakage)
  - `[9]` Li et al. (FedProx) & `[10]` Karimireddy et al. (SCAFFOLD) (Future-work algorithmic extensions)
  - `[11]` Dwork, `[12]` Abadi et al., `[13]` Bonawitz et al., `[14]` Blanchard et al., `[15]` Bhagoji et al. (Privacy & security boundaries)
- **Audit Result:** **PASS**. All literature claims match cited primary sources. No project findings are attributed to literature.

---

## 24. Future Work Audit

- **Items:** 12 structured future-work items in Section 8.2 (multi-seed evaluation, extended training horizons, decentralized preprocessing, dynamic client participation, broader datasets, FedProx/SCAFFOLD, alternative models, physical socket benchmarking, formal DP/SecAgg/poisoning defenses, physical edge hardware deployment, hierarchical topologies, cross-dataset transferability).
- **Audit Result:** **PASS**. All future-work items reflect genuine unexecuted extensions.

---

## 25. Internal and External Validity Audit

- **Internal Validity Factors:** Single seed 42, pre-fitted `StandardScaler`, PyTorch data partitioner index-slicing overhead.
- **External Validity Factors:** N-BaIoT dataset, 9 device sources, binary classification task, `SmallMLP` architecture, host CPU workstation simulation.
- **Audit Result:** **PASS**. Clear separation between internal and external validity threats is maintained in Section 7.11.

---

## 26. Numerical Precision Audit

- **Numerical Cross-Check Results:**
  - Centralized F1: Text `0.997097` vs JSON `0.9970965484478997` (Correct rounding).
  - IID FedAvg Rd 3 F1: Text `0.999430` vs JSON `0.9994299502013101` (Correct rounding).
  - Non-IID FedAvg Rd 3 F1: Text `0.975696` vs JSON `0.9756962911126662` (Correct rounding).
  - Local-Only Macro F1: Text `0.971422` vs JSON `0.9714221613024026` (Correct rounding).
  - Local-Only Weighted F1: Text `0.969684` vs JSON `0.9696841258271108` (Correct rounding).
  - Performance Margin (IID vs Non-IID F1): $0.999430 - 0.975696 = 0.023734$ (~2.37%). (Correct math).
  - Non-IID FedAvg Validation to Test F1 Delta: $0.975696 - 0.975650 = +0.000046$. (Correct math).
  - Communication Bytes: 2,059,992 bytes = 1.964561 MiB = 2.059992 MB. (Correct conversion).
- **Audit Result:** **PASS**. Zero numerical rounding or truncation errors found across text, tables, and JSONs.

---

## 27. Figure and Table Consistency Audit

- **Table 1 (Holdout Test Set):** Matches `final_test_evaluation.json`.
- **Table 2 (Validation Across Rounds):** Matches `device_non_iid_3round_seed42.json` and `iid_3round_seed42.json`.
- **Table 3 (Local-Only Device Breakdown):** Matches `final_test_evaluation.json` client breakdown.
- **Table 4 (Communication Accounting):** Matches `communication_measurement.json`.
- **Table 5 (Resource Usage & Wall Time):** Matches resource JSON files.
- **Table 6 (Validation vs Test Consistency):** Matches validation and test JSON files.
- **Figures 1–6:** Checked against PNG files in `results/processed/publication/figures/`.
- **Audit Result:** **PASS**. Complete consistency between manuscript tables/figures and raw execution artifacts.

---

## 28. Reproducibility Audit

- **Artifact Availability:** Repository `https://github.com/Moazzam9/federated-iot-ids` provides complete source code (`src/`), experiment configurations (`configs/experiment.yaml`), split specifications (`data/processed/splits/split_specification.txt`), persisted scaler (`training_standard_scaler.pkl`), and raw execution JSON logs (`results/raw/`).
- **Deterministic Flags:** `seed = 42`, Adam optimizer ($\alpha = 0.001$, batch size 256), frozen split index.
- **Audit Result:** **PASS**.

---

## 29. Claim-Strength Classification

Major manuscript statements are classified below using the three-tier claim system:
- **GREEN:** Directly supported by measured experimental evidence.
- **YELLOW:** Reasonable interpretation, but requires cautious wording or scope qualification.
- **RED:** Unsupported or materially misleading (None found).

| Location | Manuscript Claim | Classification | Supporting Evidence | Audit Finding & Recommendation |
| :--- | :--- | :---: | :--- | :--- |
| **Section 5.1 & Table 1** | Centralized test F1=0.9971, IID FedAvg F1=0.9994, Non-IID FedAvg F1=0.9757, Local-Only macro F1=0.9714. | **GREEN** | `final_test_evaluation.json` | Directly measured on frozen test set. |
| **Section 5.4 & Table 4** | 3-round cumulative parameter payload exchange total = 2,059,992 bytes (~1.96 MiB). | **GREEN** | `communication_measurement.json` | Derived directly from state dict parameter payload model (9,537 float32 params $\times$ 4 bytes). |
| **Section 5.5 & Table 5** | Peak RSS memory ranged from 610.94 MiB to 668.36 MiB; execution wall times ranged from 3,402.99 s to 16,290.68 s. | **GREEN** | Resource measurement JSON files | Directly measured via host process-tree `psutil` sampling at 0.1 s. |
| **Section 1.6 (Line 59)** | Enforces `(device, feature_hash)` grouping to prevent feature-vector data leakage across train/val/test boundaries. | **YELLOW** | `split_specification.txt` | Grouping was enforced *within each device partition*, rather than globally across devices. Stage 10 recommendation: add "within each device partition". |
| **Section 6.3 (Line 422)** | Reflecting rapid early alignment due to homogeneous local data distributions. | **YELLOW** | Section 6.3 text | Phrase "due to" claims direct causation. Stage 10 recommendation: soften "due to" to "consistent with homogeneous local data distributions." |
| **Section 6.5 & 5.5** | Controlled IID FedAvg wall time (~16,290 s) higher than Non-IID (~4,034 s) associated with PyTorch index-slicing overhead in custom data loader. | **YELLOW** | `iid_3round_seed42.json` | Reasonably attributed to index slicing in custom partitioner, but not isolated by dedicated profiling. Manuscript text is already well-guarded. |

---

## 30. Issue Severity Classification

Identified audit items are classified below by risk severity:
- **CRITICAL:** Materially invalidates or misrepresents the study (0 issues found).
- **HIGH:** Materially misleads the reader (0 issues found).
- **MEDIUM:** Recommended correction for scientific publication quality (1 minor text clarification).
- **LOW:** Minor wording or style refinement (1 minor text refinement).

| Issue ID | Severity | Location | Finding | Evidence | Stage 10 Recommended Action |
| :---: | :---: | :--- | :--- | :--- | :--- |
| **ISSUE-01** | **MEDIUM** | Section 1.6 (Line 59) | Duplicate grouping scope phrasing in Intro item 2 could be misread as global deduplication across devices. | `split_specification.txt` & Section 4.1 | Clarify line 59 by specifying: *"enforces `(device, feature_hash)` grouping within each device partition to prevent intra-device feature-vector data leakage"*. |
| **ISSUE-02** | **LOW** | Section 6.3 (Line 422) | Phrase "due to" in line 422 asserts direct causation for early IID alignment. | Section 6.3 text | Soften "due to" to *"consistent with homogeneous local data distributions"*. |

---

## 31. Recommended Stage 10 Corrections Summary

In Stage 10 (Manuscript Corrections), the following two minor refinements should be made to `manuscript/manuscript.md`:

1. **Refinement 1 (Section 1.6, Line 59):** Update contribution item 2 to explicitly state that `(device, feature_hash)` grouping is enforced *within each device partition*.
2. **Refinement 2 (Section 6.3, Line 422):** Replace "due to" with "consistent with" when referring to early IID alignment dynamics.

---

## 32. Overall Audit Conclusion

The scientific consistency audit of Publication 02 is **COMPLETE**.

**Summary Verdict:**
Every empirical value, metric table, convergence trajectory, communication volume, host resource measurement, and scope limitation reported in `manuscript/manuscript.md` is **fully consistent with the underlying implementation codebase and raw execution JSON logs**. Zero numerical errors, zero false privacy overclaims, zero fake hardware/energy measurements, and zero unsupported asymptotic convergence assertions were detected. Two minor text refinements (one MEDIUM, one LOW) are documented for Stage 10 execution.
