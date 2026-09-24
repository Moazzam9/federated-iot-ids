# Federated Learning Framework Architecture

## Overview
This document details the software architecture, data structures, control flow, and artifact persistence of the federated learning pipeline in the `federated-iot-ids` repository.

---

## 1. System Architecture Diagram

```
+-------------------------------------------------------------------+
|                        Dataset & Preprocessing                    |
|  N-BaIoT Split Loader (src/data/nbaiot_loader.py)                 |
|  StandardScaler (src/data/preprocessing.py)                       |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                     Client Partitioning Layer                     |
|  Device Partitioning (src/fl/partitioning.py: make_device_...)    |
|  IID Partitioning (src/fl/partitioning.py: make_stratified_...)   |
+-------------------------------------------------------------------+
                                  |
                                  v
+-------------------------------------------------------------------+
|                      Federated Coordinator                        |
|  Coordinator (src/fl/coordinator.py: run_federated_round)         |
|  - Broadcasts global state dict to clients                        |
|  - Triggers parallel/sequential client training                   |
|  - Collects client state dicts & sample weights                    |
|  - Performs FedAvg aggregation (src/fl/fedavg.py)                 |
+-------------------------------------------------------------------+
                                  |
            +---------------------+---------------------+
            |                                           |
            v                                           v
+-----------------------+                   +-----------------------+
|  FL Client 1          |      . . .        |  FL Client 9          |
|  (src/fl/client.py)   |                   |  (src/fl/client.py)   |
|  - Trains SmallMLP    |                   |  - Trains SmallMLP    |
|  - Returns state_dict |                   |  - Returns state_dict |
+-----------------------+                   +-----------------------+
```

---

## 2. Core Modules & Component Responsibilities

### 2.1 Data Ingestion & Preprocessing
* `src/data/nbaiot_loader.py` (`NBaIoTSplitLoader`): Streams CSV data chunks filtered by dataset split (`train`, `val`, `test`) and device name.
* `src/data/preprocessing.py` (`FittedStandardScaler`): Fits `StandardScaler` incrementally over training data chunks and transforms input features.
* `src/data/torch_data.py` (`iter_torch_batches`): Converts pandas DataFrames to PyTorch feature/label tensors in streaming batches.

### 2.2 Client Data Partitioning
* `src/fl/partitioning.py`: Defines `ClientPartition` data structures containing client IDs and row index arrays.
  * `make_device_partitions()`: Direct mapping of N-BaIoT device names to 9 logical client entities.
  * `make_stratified_iid_partitions()`: Class-stratified redistribution of training row indices across $K$ clients using seed `42`.
* `src/fl/client_data.py`: Abstracts client data loading via `DeviceClientDataSource` and `IIDClientDataSource`.

### 2.3 Local Client Execution
* `src/fl/client.py` (`train_client()`):
  * Accepts initial global model weights.
  * Instantiates local PyTorch Adam optimizer ($\alpha = 0.001$, batch size 256).
  * Executes $E$ local epochs using `src/models/trainer.py: train_one_epoch()`.
  * Returns `ClientTrainingResult` containing the updated parameter `state_dict` (detached and cloned to CPU), total local training samples, local loss, and training wall-clock duration.

### 2.4 Federated Coordination & Aggregation
* `src/fl/coordinator.py` (`run_federated_round()`):
  * Manages the lifecycle of a single federated communication round.
  * Clones the global model state to CPU before distributing to clients.
  * Executes local training for each client.
  * Collects updated client `state_dict` objects and sample counts.
  * Delegates weighted parameter aggregation to `src/fl/fedavg.py`.
  * Loads the aggregated weights back into the global model.
* `src/fl/fedavg.py` (`weighted_fedavg()`):
  * Implements element-wise sample-weighted parameter aggregation:
    $$w_{\text{global}} = \sum_{k=1}^K \left( \frac{n_k}{\sum_{j=1}^K n_j} \cdot w_k \right)$$

### 2.5 Evaluation & Measurement Utilities
* `src/models/trainer.py` (`evaluate_batches()`): Computes BCE loss, Accuracy, Precision, Recall, F1-Score, and ROC-AUC over streamed data batches.
* `experiments/scripts/measure_communication.py`: Computes theoretical model state dictionary payload bytes (float32 parameter tensor sizes).
* `experiments/scripts/measure_resource_usage.py`: Monitors host process tree resident set size (RSS memory) and wall-clock execution time at 0.1s intervals via `psutil`.

---

## 3. Artifact Hierarchy & Result Persistence

```
results/
├── raw/
│   ├── centralized/
│   │   ├── centralized_training_history.json
│   │   └── models/best_centralized_model.pt
│   ├── local_only/
│   │   ├── local_only_results.json
│   │   └── models/*.pt
│   ├── device_non_iid_3round_seed42.json
│   ├── device_non_iid_3round_seed42_models/round_00*.pt
│   ├── iid_3round_seed42.json
│   ├── iid_3round_seed42_models/round_00*.pt
│   ├── final_test_evaluation/
│   │   ├── final_test_evaluation.json
│   │   └── final_test_evaluation.csv
│   ├── communication/communication_measurement.json
│   └── resource_measurement/*.json
└── processed/
    ├── experiment_results_summary.json
    ├── analysis/*.csv
    └── publication/
        ├── publication_results.json
        ├── figures/*.png
        └── tables/*.csv
```

---

## 4. Key Architectural Constraints & Safeguards

1. **CPU Execution Scoping:** All operations strictly check and enforce `device == "cpu"`.
2. **State Dict Isolation:** State dictionaries are explicitly cloned and detached to CPU (`.detach().cpu().clone()`) to prevent shared memory pointer mutations across clients or between client and coordinator.
3. **Frozen Split Protection:** Data loaders read static index specifications, guaranteeing that evaluation splits (`validation` and `test`) remain identical regardless of partitioning strategy.
