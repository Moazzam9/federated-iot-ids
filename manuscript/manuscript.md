# Federated Learning for Intrusion Detection in IoT Networks: An Empirical Evaluation of FedAvg Performance, Resource Costs, and Communication Overhead

## Abstract
*(Section to be completed in Stage 3)*

## 1. Introduction
*(Section to be completed in Stage 3)*

## 2. Related Work
*(Section to be completed in Stage 3)*

## 3. Methodology Overview
*(Section to be completed in Stage 3)*

## 4. Experimental Setup

### 4.1 Dataset and Data Partitioning
This study evaluates intrusion detection models using the N-BaIoT (Network-Based Detection of IoT Botnet Attacks) dataset. The dataset comprises network traffic statistics collected from 9 distinct commercial IoT device types: Danmini Doorbell, Ecobee Thermostat, Ennio Doorbell, Philips B120N10 Baby Monitor, Provision PT-737E Security Camera, Provision PT-838 Security Camera, Samsung SNH-1011 N Webcam, SimpleHome XCS7-1002 WHT Security Camera, and SimpleHome XCS7-1003 WHT Security Camera. 

The primary task is binary intrusion detection, distinguishing benign network traffic (Class `0`) from malicious attack traffic (Class `1`), which encompasses Mirai and Gafgyt botnet attack vectors. Each network observation is represented by 115 continuous numerical features capturing packet arrival statistics, sizes, and jitter across five temporal decay windows (100 ms, 500 ms, 1.5 s, 10 s, 1 min) and four aggregation streams (Source-IP, Source-MAC-IP, Channel/Socket, Socket-Pair).

The complete dataset contains 7,062,606 observations. To ensure rigorous evaluation without data leakage, a frozen split specification was generated using a fixed random seed of 42. Partitioning was performed at the `(device, feature_hash)` level within each IoT device source, ensuring that identical feature-vector groups within a device were kept indivisible and assigned exclusively to one split boundary. The resulting split allocation comprises:
* **Training Set:** 4,943,824 rows (70.00%)
* **Validation Set:** 1,059,394 rows (15.00%)
* **Test Set:** 1,059,388 rows (15.00%)

Using this frozen training data, three client partitioning configurations were established:
1. **Device-Level Non-IID Partition:** The 9 physical IoT device sources are mapped directly to 9 simulated logical federated learning clients. Client dataset sizes reflect natural device traffic proportions, ranging from 248,850 training rows (Ennio Doorbell) to 769,074 training rows (Philips Baby Monitor). This setup models non-identically distributed (Non-IID) data skewed by device functionality and usage patterns.
2. **Controlled IID Partition:** The 4,943,824 training observations were redistributed deterministically across 9 simulated logical clients using class-stratified random assignment (seed 42). Each client receives approximately 549,313 to 549,315 training rows with matching benign and attack class proportions. This partition serves as an experimental control to isolate non-IID distribution effects from algorithm behavior.
3. **Local-Only Setup:** The 9 device-level logical clients train isolated local models strictly on their own local device training partition. No parameter sharing or communication occurs between clients.

The 9 clients represent simulated logical participants executing within a single-host execution environment rather than physical edge hardware nodes deployed over a wireless network.

### 4.2 Data Preprocessing
Feature standardization is performed using Scikit-Learn's `StandardScaler`. Features are transformed to zero mean and unit variance according to:
$$z = \frac{x - \mu}{\sigma}$$

In the experimental implementation, `StandardScaler` was fitted incrementally (`fit_training_scaler_incremental`) strictly over the combined 4,943,824 training rows prior to federated partitioning, and persisted to disk (`data/processed/preprocessing/training_standard_scaler.pkl`). All validation and test partitions were transformed using these fixed training mean ($\mu$) and standard deviation ($\sigma$) vectors without refitting.

*Methodological Note & Federated Limitation:* Fitting the scaler centrally across pooled training data was selected to ensure uniform feature scaling across centralized, local-only, and federated experiments. However, because global feature statistics ($\mu, \sigma$) were computed centrally prior to client partitioning, this preprocessing pipeline does not represent a fully decentralized edge pipeline. In an actual edge deployment, computing global feature statistics across isolated participants would require a federated analytics protocol.

### 4.3 Model Architecture
All experiments evaluate a lightweight feed-forward Multi-Layer Perceptron (`SmallMLP`) implemented in PyTorch (`src/models/mlp.py`). The architecture consists of:
* **Input Layer:** 115 continuous features
* **Hidden Layer 1:** 64 linear units followed by Rectified Linear Unit (ReLU) activation
* **Hidden Layer 2:** 32 linear units followed by Rectified Linear Unit (ReLU) activation
* **Output Layer:** 1 linear unit followed by Sigmoid activation, producing an estimated probability $p \in [0, 1]$ of attack traffic.

The network contains exactly **9,537 trainable parameters**, calculated as:
$$\text{Layer 1 Weights \& Biases: } (115 \times 64) + 64 = 7,424$$
$$\text{Layer 2 Weights \& Biases: } (64 \times 32) + 32 = 2,080$$
$$\text{Output Layer Weights \& Biases: } (32 \times 1) + 1 = 33$$
$$\text{Total Parameters: } 7,424 + 2,080 + 33 = 9,537$$

Training minimizes Binary Cross-Entropy loss over batch predictions:
$$\mathcal{L}_{BCE} = -\frac{1}{N} \sum_{i=1}^N \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]$$

### 4.4 Training Configuration
Models were trained using the Adam optimizer with a fixed learning rate of $\alpha = 0.001$ and a batch size of 256 samples. Random seeds were fixed to 42 across NumPy and PyTorch. All computations were executed on CPU.

*Configured Maximums vs. Executed Runs:*
While project configuration files (`configs/experiment.yaml`) specify upper bounds allowing up to 10 epochs or rounds, the empirical experiments completed and reported in this manuscript executed the following parameters:
* **Centralized Baseline:** 3 full training epochs over the 4,943,824 pooled training rows.
* **Local-Only Baseline:** 3 training epochs for each of the 9 independent local device client models.
* **Federated Learning (FedAvg):** 3 communication rounds, with $E = 1$ local epoch per round for each client.
* **Client Participation:** Full client participation in every round ($C = 1.0$, 9 out of 9 clients participating).

### 4.5 Federated Learning Procedure
Federated experiments execute the standard Federated Averaging (FedAvg) algorithm (`src/fl/coordinator.py`). For each communication round $r \in \{1, 2, 3\}$:
1. **Global Model Broadcast:** The central coordinator serializes the global model parameter state dictionary $W^{r-1}$ and distributes a copy to all 9 participating clients.
2. **Local Training:** Each client $k \in \{1, \dots, 9\}$ instantiates a local model initialized with $W^{r-1}$ and trains locally for $E = 1$ epoch over its assigned training subset $D_k$ ($n_k = |D_k|$) using Adam ($\alpha = 0.001$, batch size 256).
3. **State Upload:** Upon local training completion, each client returns its updated parameter state dictionary $W_k^r$ and sample count $n_k$ to the coordinator.
4. **Weighted Aggregation:** The coordinator computes the updated global parameter state dictionary $W^r$ via sample-weighted parameter averaging:
   $$W^r = \sum_{k=1}^{K} \frac{n_k}{N} W_k^r \quad \text{where } N = \sum_{k=1}^K n_k = 4,943,824, \; K = 9$$
5. **Global Model Update:** The global model parameters are updated with $W^r$ to complete the round.

Transmission involves full model parameter state dictionaries (`state_dict`), not loss gradients.

### 4.6 Baselines
Four experimental conditions were evaluated to isolate data distribution and collaboration effects:
1. **Centralized Baseline:** Model trained on pooled global training data. Represents the empirical upper bound for model learning capacity without data distribution boundaries.
2. **Local-Only Baseline:** 9 isolated models trained independently on single-device data. Represents performance when clients do not participate in collaborative training.
3. **Device-Level Non-IID FedAvg:** FedAvg executed across the 9 natural device partitions. Measures collaborative learning performance under real-world device distribution heterogeneity.
4. **Controlled IID FedAvg:** FedAvg executed across 9 class-stratified IID partitions. Isolates FedAvg algorithmic convergence from device non-IID skew.

### 4.7 Evaluation Protocol
Model performance is evaluated across six metrics: Loss, Accuracy, Precision, Recall, F1-Score, and ROC-AUC. Classification decisions use a fixed probability threshold of 0.5 ($p \ge 0.5 \rightarrow \text{Attack}$).

*Evaluation Scope Distinction:*
* **Global Test Scope:** The Centralized model, IID FedAvg global model, and Device Non-IID FedAvg global model are evaluated on the complete, frozen global test set (1,059,388 rows across all 9 devices).
* **Local-Only Test Scope:** Each Local-Only model is evaluated strictly on the test partition belonging to its corresponding device (sample sizes ranging from 53,325 to 164,801 rows; summing to 1,059,388 rows total). Both macro-averages (equal weight per client) and sample-weighted averages are reported. Validation and test sets were evaluated strictly using the pre-fitted training scaler without refitting.

### 4.8 Communication Payload Accounting
Communication cost is evaluated using a model parameter state dictionary payload accounting model (`experiments/scripts/measure_communication.py`). 

* **State Dict Payload Size:** The `SmallMLP` model contains 9,537 float32 parameters (4 bytes per parameter), yielding a tensor weight payload of **38,148 bytes** (~37.25 KiB / 0.03815 MB).
* **Per-Client Per-Round Exchange:** Each participating client downloads 1 global model state dictionary (38,148 bytes) and uploads 1 updated model state dictionary (38,148 bytes), totaling **76,296 bytes** (~74.51 KiB) per client per round.
* **Per-Round Total (9 Clients):** $9 \times 76,296 = 686,664 \text{ bytes}$ (~0.655 MiB / 0.687 MB) per round.
* **3-Round Experiment Total:** $3 \times 686,664 = 2,059,992 \text{ bytes}$ (~1.96 MiB / 2.06 MB).

*Explicit Accounting Scope:* These figures represent theoretical model tensor payload accounting. They do NOT represent measured network socket traffic, nor do they include TCP/IP packet headers, TLS/SSL encryption overhead, HTTP/gRPC frame headers, serialization/deserialization overhead, payload compression, network latency, or packet retransmission.

### 4.9 Resource Measurement
Resource overhead was measured during experiment execution using process-tree Resident Set Size (RSS) monitoring via Python `psutil` (v7.2.2) sampled at 0.1-second intervals (`experiments/scripts/measure_resource_usage.py`).

* **Centralized Baseline (3 epochs):** Peak RSS = 610.94 MiB (640,614,400 bytes), Wall Time = 5,018.94 s (~83.6 min).
* **Local-Only Baseline (3 epochs):** Peak RSS = 610.95 MiB (640,630,784 bytes), Wall Time = 3,402.99 s (~56.7 min).
* **Device Non-IID FedAvg (3 rounds):** Peak RSS = 621.54 MiB (651,730,944 bytes), Wall Time = 4,034.84 s (~67.2 min).
* **IID FedAvg (3 rounds):** Peak RSS = 668.36 MiB (700,821,504 bytes), Wall Time = 16,290.68 s (~271.5 min).

*Measurement Scope:* Reported peak RSS reflects host RAM allocated to the Python process tree on CPU execution. Runtimes and memory usage represent single-host simulation performance and do not measure edge device battery drain, physical hardware RAM constraints, or energy consumption.

### 4.10 Reproducibility Statement
To ensure full reproducibility, the repository includes fixed random seed definitions (seed 42), frozen split index specifications (`data/processed/splits/split_specification.txt`), persisted StandardScaler binaries (`training_standard_scaler.pkl`), complete experiment configurations (`configs/experiment.yaml`), and raw experiment output logs (`results/raw/`). The complete codebase and results packaging scripts are maintained at `https://github.com/Moazzam9/federated-iot-ids`.

---

## 5. Results
*(Section to be completed in Stage 3)*

## 6. Discussion
*(Section to be completed in Stage 4)*

## 7. Limitations
*(Section to be completed in Stage 4)*

## 8. Conclusion
*(Section to be completed in Stage 4)*
