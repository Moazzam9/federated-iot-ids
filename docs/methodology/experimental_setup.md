# Experimental Setup Documentation

## Overview
This document provides a comprehensive technical overview of the experimental setup, data partitioning, model architecture, training configuration, evaluation protocol, communication accounting, and resource monitoring for the study (`federated-iot-ids`).

---

## 1. Dataset & Data Partitioning

### Dataset Summary
* **Name:** N-BaIoT (Network-Based Detection of IoT Botnet Attacks)
* **Total Observations:** 7,062,606 rows
* **Features:** 115 continuous numerical features (packet statistics across 5 time windows and 4 stream aggregations)
* **Task:** Binary classification (`0` = Benign, `1` = Attack: Mirai & Gafgyt)
* **Device Sources (9 Devices):**
  1. Danmini Doorbell
  2. Ecobee Thermostat
  3. Ennio Doorbell
  4. Philips B120N10 Baby Monitor
  5. Provision PT-737E Security Camera
  6. Provision PT-838 Security Camera
  7. Samsung SNH-1011 N Webcam
  8. SimpleHome XCS7-1002 WHT Security Camera
  9. SimpleHome XCS7-1003 WHT Security Camera

### Frozen Dataset Split Specification
Partitions were generated using random seed `42` at the `(device, feature_hash)` group level to prevent identical feature vectors within a device from leaking across split boundaries.

* **Training Split:** 4,943,824 rows (70.00%)
* **Validation Split:** 1,059,394 rows (15.00%)
* **Test Split:** 1,059,388 rows (15.00%)
* **Total:** 7,062,606 rows (100.00%)

### Client Partition Strategies
1. **Device-Level Non-IID Partition:** 9 logical clients corresponding to the 9 natural N-BaIoT IoT devices. Training sample counts per client:
   * `Danmini_Doorbell`: 712,809 rows
   * `Ecobee_Thermostat`: 585,113 rows
   * `Ennio_Doorbell`: 248,850 rows
   * `Philips_B120N10_Baby_Monitor`: 769,074 rows
   * `Provision_PT_737E_Security_Camera`: 579,782 rows
   * `Provision_PT_838_Security_Camera`: 585,824 rows
   * `Samsung_SNH_1011_N_Webcam`: 262,655 rows
   * `SimpleHome_XCS7_1002_WHT_Security_Camera`: 604,139 rows
   * `SimpleHome_XCS7_1003_WHT_Security_Camera`: 595,578 rows
2. **Controlled IID Partition:** Training split (4,943,824 rows) redistributed across 9 clients using class-stratified random assignment (seed `42`). Each client receives ~549,313–549,315 rows with matching benign/attack proportions.
3. **Local-Only Partition:** The 9 device clients train independent models using only their respective device training data.

*Interpretation Note:* Clients are simulated logical entities in a single-host execution environment, not physical hardware edge nodes.

---

## 2. Data Preprocessing

* **Scaler Type:** `StandardScaler` (Scikit-Learn)
* **Fitting Scope:** Fitted incrementally (`fit_training_scaler_incremental`) over the combined 4.94M training set (`training_standard_scaler.pkl`).
* **Transformation Scope:** Validation and test sets are transformed using pre-fitted training mean and variance vectors without refitting.
* **Methodological Limitation:** Because feature statistics were computed centrally prior to client partitioning, this preprocessing pipeline does not represent a fully decentralized edge pipeline.

---

## 3. Model Architecture (`SmallMLP`)

Defined in `src/models/mlp.py`:

```
Input (115) --> Dense(64) --> ReLU --> Dense(32) --> ReLU --> Dense(1) --> Sigmoid
```

* **Trainable Parameter Count:** 9,537 float32 parameters
  * Layer 1: $(115 \times 64) + 64 = 7,424$
  * Layer 2: $(64 \times 32) + 32 = 2,080$
  * Output Layer: $(32 \times 1) + 1 = 33$
* **Loss Function:** Binary Cross-Entropy (`BCELoss`)

---

## 4. Training Configurations (Executed Runs)

* **Optimizer:** Adam ($\alpha = 0.001$)
* **Batch Size:** 256
* **Device:** CPU
* **Seed:** 42
* **Executed Run Parameters:**
  * Centralized Baseline: 3 epochs
  * Local-Only Baseline: 3 epochs per client
  * Federated Learning (FedAvg): 3 rounds, 1 local epoch per round, full client participation (9/9 clients per round)

---

## 5. Federated Learning Algorithm (FedAvg)

Implemented in `src/fl/coordinator.py` and `src/fl/fedavg.py`:

1. **Broadcast:** Server sends global state dict $W^{r-1}$ to all 9 clients.
2. **Local Training:** Each client initializes local model with $W^{r-1}$ and trains for 1 epoch over local data $D_k$.
3. **Upload:** Each client sends updated state dict $W_k^r$ and sample count $n_k$ to server.
4. **Aggregation:** Server updates global model via sample-weighted parameter averaging:
   $$W^r = \sum_{k=1}^K \frac{n_k}{N} W_k^r$$

---

## 6. Evaluation Protocols & Scope Distinction

* **Metrics:** Loss, Accuracy, Precision, Recall, F1-Score, ROC-AUC (decision threshold = 0.5).
* **Test Scope Difference:**
  * **Global Test Scope:** Centralized and FedAvg global models are evaluated on the global test set (1,059,388 rows).
  * **Local-Only Test Scope:** Local-Only models are evaluated on their respective device-specific test subsets (summing to 1,059,388 rows total).

---

## 7. Communication Accounting

Calculated by `experiments/scripts/measure_communication.py`:

* **Model State Dict Size:** 38,148 bytes (~37.25 KiB)
* **Per Client Per Round:** 76,296 bytes (Download + Upload)
* **Per Round (9 Clients):** 686,664 bytes (~0.655 MiB)
* **3-Round Total:** 2,059,992 bytes (~1.96 MiB)

*Accounting Scope Note:* Represents model tensor payload calculations. Excludes network headers, TLS encryption, serialization overhead, compression, and network latency.

---

## 8. Resource Measurements

Measured by `experiments/scripts/measure_resource_usage.py` (`psutil` 0.1s RSS sampling):

* **Centralized (3 epochs):** Peak RSS = 610.94 MiB, Wall Time = 5,018.94 s
* **Local-Only (3 epochs):** Peak RSS = 610.95 MiB, Wall Time = 3,402.99 s
* **Device Non-IID FedAvg (3 rounds):** Peak RSS = 621.54 MiB, Wall Time = 4,034.84 s
* **IID FedAvg (3 rounds):** Peak RSS = 668.36 MiB, Wall Time = 16,290.68 s

*Resource Scope Note:* Process tree RSS memory on host CPU. Does not measure edge hardware RAM limits or physical energy consumption.
