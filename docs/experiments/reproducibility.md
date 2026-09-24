# Reproducibility Guide — Publication 02

This document provides complete reproducibility documentation for the research study:

> **Title:** *Federated Learning for Resource-Aware Intrusion Detection in IoT Networks: An Empirical Study of IID and Device-Level Non-IID Data*  
> **Repository:** `federated-iot-ids` (`https://github.com/Moazzam9/federated-iot-ids.git`)

---

## 1. Reproducibility Overview

This guide enables a technically competent researcher to understand the computational environment, acquire the dataset, reproduce data split and preprocessing procedures, reconstruct experimental conditions, execute baseline and federated model training scripts, perform evaluation, run resource and communication measurement tools, generate publication tables/figures, and locate frozen research artifacts.

The study compares four experimental conditions on the N-BaIoT intrusion detection benchmark:
1. **Centralized Baseline:** Global model trained on pooled training data (4,943,824 rows, 3 epochs).
2. **Local-Only Baseline:** 9 isolated device models trained independently (3 epochs per client).
3. **Device-Level Non-IID FedAvg:** Federated Averaging across 9 natural device traffic clients (3 rounds, $E=1$ local epoch/round, seed 42).
4. **Controlled IID FedAvg:** Federated Averaging across 9 class-stratified IID partitions (3 rounds, $E=1$ local epoch/round, seed 42).

All experiments evaluate a lightweight 9,537-parameter feed-forward neural network (`SmallMLP`: 115 $\rightarrow$ 64 $\rightarrow$ 32 $\rightarrow$ 1) implemented in PyTorch under CPU-only execution.

---

## 2. Hardware and Software Environment

The primary empirical evaluations reported in this study were executed in the following verified system environment:

* **Operating System:** Microsoft Windows 10 Pro 64-bit (v10.0.19045)
* **Python Interpreter:** Python 3.12.10 64-bit (`MSC v.1943 64 bit AMD64`)
* **Host Processor (CPU):** Intel(R) Core(TM) i7-6820HQ CPU @ 2.70 GHz (4 Physical Cores / 8 Logical Processors)
* **Host RAM:** ~17 GB RAM (17,042 MB physical memory)
* **Graphics Hardware (GPU):** None used; all computations were executed strictly on CPU.
* **Version Control System:** Git for Windows (v2.50.1.windows.1)
* **Virtual Environment Location:** `D:\federated-iot-ids\.venv`

---

## 3. Repository Setup

To set up the repository environment locally:

```powershell
# Clone the repository
git clone https://github.com/Moazzam9/federated-iot-ids.git
cd federated-iot-ids

# Create virtual environment (Python 3.12)
python -m venv .venv

# Activate virtual environment (Windows PowerShell)
.\.venv\Scripts\Activate.ps1
```

---

## 4. Dependency Installation

The complete dependency specification is maintained in `requirements.txt`. Key package versions are pinned as follows:

```text
numpy==2.5.3
pandas==3.0.5
PyYAML==6.0.3
torch==2.14.0
pytest==9.1.1
scikit-learn==1.9.1
psutil==7.2.2
matplotlib==3.11.2
```

To install all verified dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

---

## 5. Dataset Acquisition

The study uses the publicly available **N-BaIoT (Network-Based Detection of IoT Botnet Attacks)** dataset:

* **Primary Literature Reference:** Y. Meidan et al., *"N-BaIoT—Network-based detection of IoT botnet attacks using deep autoencoders,"* IEEE Pervasive Computing, vol. 17, no. 3, pp. 12–22, 2018. DOI: 10.1109/MPRV.2018.03367731.
* **Dataset Scope:** 7,062,606 total observations across 89 CSV files spanning 9 commercial IoT device types:
  1. `Danmini_Doorbell` (712,809 train / 152,744 test)
  2. `Ecobee_Thermostat` (585,113 train / 125,381 test)
  3. `Ennio_Doorbell` (248,850 train / 53,325 test)
  4. `Philips_B120N10_Baby_Monitor` (769,074 train / 164,801 test)
  5. `Provision_PT_737E_Security_Camera` (579,782 train / 124,239 test)
  6. `Provision_PT_838_Security_Camera` (585,824 train / 125,533 test)
  7. `Samsung_SNH_1011_N_Webcam` (262,655 train / 56,283 test)
  8. `SimpleHome_XCS7_1002_WHT_Security_Camera` (604,139 train / 129,458 test)
  9. `SimpleHome_XCS7_1003_WHT_Security_Camera` (595,578 train / 127,624 test)

```text
MANUAL ACTION REQUIRED:
Download the 89 uncompressed N-BaIoT device CSV files from the repository/archive source and place them in data/raw/ or update the root path configuration in configs/experiment.yaml.
```

---

## 6. Dataset Verification & Inventory

To verify the raw dataset structure and row counts after placing CSV files:

```powershell
python -m experiments.scripts.inventory_nbaiot_rows
```

Expected output:
* 89 source CSV files detected.
* Total observations: 7,062,606 rows.
* Features per row: 115 numerical features + binary label.

---

## 7. Split Construction & Duplicate Audit

The study uses a duplicate-aware frozen split specification (`(device, feature_hash)` grouping within each device partition, seed 42):

* **Training Set:** 4,943,824 rows (70.00%)
* **Validation Set:** 1,059,394 rows (15.00%)
* **Test Set:** 1,059,388 rows (15.00%)

The frozen split assignments are persisted in `data/processed/splits/split_specification.txt`. To inspect split properties or audit duplicate structure:

```powershell
python -m experiments.scripts.audit_duplicate_structure
```

*Note on Frozen Artifacts:* Split index arrays are stored in `data/processed/splits/`. They are frozen research artifacts and should not be re-generated for standard experiment replication.

---

## 8. Data Preprocessing

Feature standardization is performed using Scikit-Learn's `StandardScaler`:
$$z = \frac{x - \mu}{\sigma}$$

* **Methodological Protocol:** `StandardScaler` was fitted incrementally (`fit_training_scaler_incremental`) over the combined 4,943,824 global training rows prior to federated partitioning.
* **Persisted Binary:** `data/processed/preprocessing/training_standard_scaler.pkl`
* **Reuse:** All validation and test set evaluations reuse these fixed training mean ($\mu$) and standard deviation ($\sigma$) vectors without refitting.

To verify the persisted preprocessing scaler:

```powershell
python -m experiments.scripts.verify_training_scaler
```

---

## 9. Centralized Baseline Experiment

To execute the Centralized baseline training (3 epochs over 4,943,824 training samples):

```powershell
python -m experiments.scripts.run_centralized --epochs 3
```

Outputs written to:
* Model Checkpoint: `results/raw/centralized/models/best_centralized_model.pt`
* Training History JSON: `results/raw/centralized/centralized_training_history.json`

---

## 10. Local-Only Baseline Experiment

To execute isolated Local-Only baseline training (3 epochs independently for each of the 9 device clients):

```powershell
python -m experiments.scripts.run_local_only --epochs 3 --seed 42
```

Outputs written to:
* Model Checkpoints: `results/raw/local_only/models/{device_name}.pt`
* Results JSON: `results/raw/local_only/local_only_results.json`

---

## 11. Device-Level Non-IID FedAvg Experiment

To execute Federated Averaging across the 9 natural device-level clients (3 rounds, $E=1$ local epoch/round, Adam $\alpha=0.001$, batch size 256, seed 42):

```powershell
python -m experiments.scripts.run_multiround_federated --partition device_non_iid --rounds 3 --local-epochs 1 --batch-size 256 --learning-rate 0.001 --seed 42 --output-prefix device_non_iid_3round_seed42
```

Outputs written to:
* Round Checkpoints: `results/raw/device_non_iid_3round_seed42_models/round_00{1,2,3}.pt`
* Experiment Log JSON: `results/raw/device_non_iid_3round_seed42.json`

---

## 12. Controlled IID FedAvg Experiment

To execute Federated Averaging across the 9 controlled class-stratified IID partitions (3 rounds, $E=1$ local epoch/round, Adam $\alpha=0.001$, batch size 256, seed 42):

```powershell
python -m experiments.scripts.run_multiround_federated --partition iid --rounds 3 --local-epochs 1 --batch-size 256 --learning-rate 0.001 --seed 42 --output-prefix iid_3round_seed42
```

Outputs written to:
* Round Checkpoints: `results/raw/iid_3round_seed42_models/round_00{1,2,3}.pt`
* Experiment Log JSON: `results/raw/iid_3round_seed42.json`

---

## 13. Final Holdout Evaluation

To execute evaluation on final holdout test sets across all trained models:

```powershell
python -m experiments.scripts.evaluate_final_test_set
```

* **Evaluation Scope:**
  - Centralized, Device Non-IID FedAvg, and IID FedAvg models are evaluated on the complete global test set (1,059,388 rows).
  - Local-Only models are evaluated on their respective device-specific test subsets (summing to 1,059,388 rows).
* **Outputs:**
  - `results/raw/final_test_evaluation/final_test_evaluation.json`
  - `results/raw/final_test_evaluation/final_test_evaluation.csv`

---

## 14. Communication Payload Accounting

Model parameter payload accounting calculates tensor exchange volume based on state dictionary size (`state_dict`):

* **Model State Dict Size:** 9,537 float32 parameters $\times$ 4 bytes = **38,148 bytes** (~37.25 KiB).
* **Per Client / Round:** 1 download + 1 upload = **76,296 bytes** (~74.51 KiB).
* **Per Round (9 Clients):** $9 \times 76,296 = \mathbf{686,664 \text{ bytes}}$ (~0.655 MiB).
* **3-Round Total:** $3 \times 686,664 = \mathbf{2,059,992 \text{ bytes}}$ (~1.96 MiB / 2.06 MB).

To verify communication accounting metrics:

```powershell
python -m experiments.scripts.measure_communication
```

Output written to: `results/raw/communication/communication_measurement.json`.

---

## 15. Resource Usage Monitoring

Process-tree Resident Set Size (RSS memory) and execution wall-clock time are monitored using Python `psutil` at 0.1-second sampling intervals:

```powershell
# Example: Wrapping an experiment execution with resource monitoring
python -m experiments.scripts.measure_resource_usage --output results/raw/resource_measurement/example_run.json -- python -m experiments.scripts.run_centralized --epochs 3
```

Recorded baseline resource outputs:
* Centralized Baseline: Peak RSS = **610.94 MiB**, Wall Time = **5,018.94 s** (~83.6 min)
* Local-Only Baseline: Peak RSS = **610.95 MiB**, Wall Time = **3,402.99 s** (~56.7 min)
* Device Non-IID FedAvg: Peak RSS = **621.54 MiB**, Wall Time = **4,034.84 s** (~67.2 min)
* Controlled IID FedAvg: Peak RSS = **668.36 MiB**, Wall Time = **16,290.68 s** (~271.5 min)

---

## 16. Results Summarization & Figure/Table Generation

To process raw experiment JSON logs and generate publication-ready tables and figures:

```powershell
# Step 1: Build results summary JSON
python -m experiments.scripts.build_results_summary

# Step 2: Build publication tables (Markdown/CSV) and figures (PNG)
python -m experiments.scripts.build_publication_results
```

Generated publication artifacts:
* Tables: `results/processed/publication/tables/` (`table_1_final_test_results.md`, `table_2_fedavg_convergence.md`, `table_3_local_only_by_device.md`, `table_4_communication_accounting.md`, `table_5_resource_usage.md`, `table_6_validation_vs_test.md`).
* Figures: `results/processed/publication/figures/` (`fedavg_f1_convergence.png`, `fedavg_roc_auc_convergence.png`, `local_only_f1_by_device.png`, `communication_payload.png`, `experiment_runtime.png`, `peak_memory_usage.png`).

---

## 17. Automated Validation Test Suite

The project includes unit and integration tests covering model architecture, FedAvg aggregation, client data slicing, preprocessing scaling, and IID partitioning:

```powershell
# Run all unit and integration tests
.\.venv\Scripts\python.exe -m pytest --basetemp=build/pytest_tmp -q
```

Expected output:
```text
100 passed in ~80-120 seconds
```

---

## 18. Expected Artifacts and Git Tracking Policy

| Directory / File | Description | Tracking Status in Git |
| :--- | :--- | :--- |
| `src/` | Complete Python source code | Tracked |
| `configs/experiment.yaml` | Primary experiment configuration | Tracked |
| `data/processed/splits/split_specification.txt` | Frozen N-BaIoT split specification | Tracked |
| `data/processed/preprocessing/training_standard_scaler.pkl` | Pre-fitted StandardScaler binary | Tracked |
| `manuscript/manuscript.md` | Full research paper manuscript | Tracked |
| `docs/` | Scientific audit and validation reports | Tracked |
| `results/raw/` | Raw JSON execution logs and model checkpoints | Ignored (`.gitignore`); tracked metadata files exempted |
| `results/processed/publication/` | Final summary tables and figures | Tracked |
| `data/raw/*` | Raw 89 N-BaIoT device CSV files | Ignored (`.gitignore`); acquired separately |

---

## 19. Methodological Scope & Reproducibility Limitations

Researchers replicating or extending this study should note the following explicit boundaries:

1. **Dataset Acquisition:** N-BaIoT CSV files are not hosted in the Git repository and must be acquired separately by the user.
2. **Hardware Dependence:** Wall-clock runtimes and process-tree RSS memory reflect execution on an Intel i7-6820HQ CPU workstation; exact runtimes and RAM usage will vary across host hardware.
3. **Single Seed Execution:** The reported empirical results reflect execution under a single fixed seed (`seed = 42`) across 3 training rounds/epochs; cross-seed variability is not measured.
4. **Theoretical Communication Payload:** Communication volume figures reflect model tensor weight payload accounting (`state_dict` size), not physical network socket traffic or transport protocol framing.
5. **Pre-fitted Preprocessing:** `StandardScaler` was fitted globally on pooled training data prior to partitioning. Obtaining global feature statistics in edge deployments would require federated analytics.
6. **Simulated Participants:** Clients represent logical data partitions executing within a single-host CPU simulation rather than physical edge hardware nodes over wireless interfaces.
