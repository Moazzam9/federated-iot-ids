from __future__ import annotations

import json
import sys
import time
from pathlib import Path

# ----------------------------------------------------------------------
# Make the project root importable when this script is executed directly.
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


import joblib
import torch
from torch import nn

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.data.torch_data import iter_torch_batches
from src.fl.client_data import DeviceClientDataSource
from src.fl.coordinator import run_federated_round
from src.fl.partitioning import make_device_partitions
from src.models.mlp import SmallMLP
from src.models.trainer import evaluate_batches, set_random_seed


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

DATA_ROOT = Path(r"D:\federated-iot-temp")

SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "training_standard_scaler.pkl"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
    / "raw"
)

MODEL_PATH = (
    RESULTS_DIR
    / "device_non_iid_federated_pilot_global_model.pt"
)

RESULT_JSON_PATH = (
    RESULTS_DIR
    / "device_non_iid_federated_pilot.json"
)


# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------

SEED = 42

NUM_CLIENTS = 9

LOCAL_EPOCHS = 1

BATCH_SIZE = 256

LEARNING_RATE = 0.001

DEVICE = "cpu"

EXPECTED_TRAINING_ROWS = 4_943_824

EXPECTED_VALIDATION_ROWS = 1_059_394

EXPECTED_FEATURE_COUNT = 115

EXPECTED_PARAMETER_COUNT = 9_537


# ----------------------------------------------------------------------
# N-BaIoT logical device clients
# ----------------------------------------------------------------------

CLIENT_IDS = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]


# ----------------------------------------------------------------------
# Validation adapter
# ----------------------------------------------------------------------

def validation_tuple_batches(
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
    batch_size: int,
):
    """
    Adapt BatchData objects produced by iter_torch_batches()
    into (features, labels) tuples expected by evaluate_batches().
    """

    for batch in iter_torch_batches(
        loader=loader,
        scaler=scaler,
        split="validation",
        batch_size=batch_size,
        device=DEVICE,
    ):
        yield batch.features, batch.labels


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def main() -> None:

    print("=" * 72)
    print("DEVICE NON-IID FEDAVG PILOT")
    print("=" * 72)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset root: {DATA_ROOT}")
    print(f"Split root:   {SPLIT_ROOT}")
    print(f"Scaler path:  {SCALER_PATH}")
    print(f"Model path:   {MODEL_PATH}")
    print()

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    set_random_seed(SEED)

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ------------------------------------------------------------------
    # 1. Create and validate N-BaIoT loader
    # ------------------------------------------------------------------

    print("1. CREATING N-BAIoT LOADER")

    loader = NBaIoTSplitLoader(
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
    )

    loader.validate_source_indexes()

    train_rows = loader.total_rows_for_split(
        "train"
    )

    validation_rows = loader.total_rows_for_split(
        "validation"
    )

    if train_rows != EXPECTED_TRAINING_ROWS:
        raise RuntimeError(
            "Training row-count mismatch. "
            f"Expected {EXPECTED_TRAINING_ROWS:,}, "
            f"got {train_rows:,}."
        )

    if validation_rows != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(
            "Validation row-count mismatch. "
            f"Expected {EXPECTED_VALIDATION_ROWS:,}, "
            f"got {validation_rows:,}."
        )

    print("Source split indexes: PASS")
    print(
        f"Training rows:   {train_rows:,}"
    )
    print(
        f"Validation rows: {validation_rows:,}"
    )
    print()

    # ------------------------------------------------------------------
    # 2. Validate device partitions
    # ------------------------------------------------------------------

    print("2. VALIDATING DEVICE PARTITIONS")

    partitions = make_device_partitions(
        CLIENT_IDS
    )

    if not isinstance(partitions, dict):
        raise RuntimeError(
            "make_device_partitions() must return a dictionary."
        )

    if len(partitions) != NUM_CLIENTS:
        raise RuntimeError(
            f"Expected {NUM_CLIENTS} device partitions, "
            f"but received {len(partitions)}."
        )

    partition_ids = list(
        partitions.keys()
    )

    if partition_ids != CLIENT_IDS:
        raise RuntimeError(
            "Device partition IDs do not match the expected "
            "N-BaIoT device client IDs."
        )

    if len(set(partition_ids)) != NUM_CLIENTS:
        raise RuntimeError(
            "Duplicate device client IDs detected."
        )

    for client_id in CLIENT_IDS:

        partition = partitions[client_id]

        if partition.client_id != client_id:
            raise RuntimeError(
                f"Partition identity mismatch for {client_id}."
            )

        if partition.row_indices.dtype.name != "int64":
            raise RuntimeError(
                f"Unexpected row-index dtype for {client_id}."
            )

        if len(partition.row_indices) != 0:
            raise RuntimeError(
                f"Device partition {client_id} unexpectedly "
                "contains copied row indices."
            )

    print(
        f"Device partitions: {len(partitions)}"
    )

    print("Device/client mapping: PASS")
    print("Identity-only partition design: PASS")
    print()

    # ------------------------------------------------------------------
    # 3. Load persisted training-only scaler
    # ------------------------------------------------------------------

    print("3. LOADING TRAINING SCALER")

    if not SCALER_PATH.is_file():
        raise FileNotFoundError(
            f"Training scaler was not found:\n{SCALER_PATH}"
        )

    scaler = joblib.load(
        SCALER_PATH
    )

    if not isinstance(
        scaler,
        FittedStandardScaler,
    ):
        raise TypeError(
            "Unexpected scaler type: "
            f"{type(scaler).__name__}. "
            "Expected FittedStandardScaler."
        )

    print("Training-only scaler: PASS")
    print()

    # ------------------------------------------------------------------
    # 4. Create device client data source
    # ------------------------------------------------------------------

    print("4. CREATING DEVICE CLIENT DATA SOURCE")

    client_data_source = DeviceClientDataSource(
        loader
    )

    client_training_samples: dict[str, int] = {}

    for client_id in CLIENT_IDS:

        count = (
            client_data_source.count_training_samples(
                client_id
            )
        )

        if count <= 0:
            raise RuntimeError(
                f"Client {client_id!r} has no training samples."
            )

        client_training_samples[
            client_id
        ] = count

        print(
            f"{client_id}: {count:,}"
        )

    total_training_samples = sum(
        client_training_samples.values()
    )

    if total_training_samples != EXPECTED_TRAINING_ROWS:
        raise RuntimeError(
            "Global training row mismatch. "
            f"Expected {EXPECTED_TRAINING_ROWS:,}, "
            f"got {total_training_samples:,}."
        )

    print()
    print(
        f"Total training samples: "
        f"{total_training_samples:,}"
    )

    print("Client sample counts: PASS")
    print()

    # ------------------------------------------------------------------
    # 5. Create model
    # ------------------------------------------------------------------

    print("5. CREATING GLOBAL MODEL")

    global_model = SmallMLP(
        input_dim=EXPECTED_FEATURE_COUNT
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in global_model.parameters()
        if parameter.requires_grad
    )

    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "Unexpected model parameter count. "
            f"Expected {EXPECTED_PARAMETER_COUNT:,}, "
            f"got {parameter_count:,}."
        )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    print("Model architecture: PASS")
    print()

    # ------------------------------------------------------------------
    # 6. Load existing completed model OR run FedAvg
    # ------------------------------------------------------------------

    fedavg_was_run = False

    round_result = None

    measured_round_time = None

    if MODEL_PATH.is_file():

        # --------------------------------------------------------------
        # Existing model found.
        #
        # This is important because the previous run already completed
        # all nine clients and saved the global model before validation
        # failed.
        #
        # Therefore we DO NOT retrain.
        # --------------------------------------------------------------

        print("=" * 72)
        print("EXISTING GLOBAL MODEL FOUND")
        print("=" * 72)

        print(
            "The FedAvg training phase was already completed."
        )

        print(
            "Skipping retraining and proceeding directly "
            "to validation."
        )

        print(
            f"Existing model: {MODEL_PATH}"
        )

        print()

    else:

        # --------------------------------------------------------------
        # No model exists, so this is a fresh experiment.
        # --------------------------------------------------------------

        print("=" * 72)
        print("STARTING FEDAVG ROUND")
        print("=" * 72)

        print("Partition type: device-based Non-IID")
        print("Clients:         9")
        print("Local epochs:    1")
        print("Batch size:      256")
        print("Learning rate:   0.001")
        print("Device:          CPU")
        print()

        round_start = time.perf_counter()

        round_result = run_federated_round(
            global_model=global_model,
            loader=loader,
            scaler=scaler,
            client_ids=CLIENT_IDS,
            local_epochs=LOCAL_EPOCHS,
            batch_size=BATCH_SIZE,
            learning_rate=LEARNING_RATE,
            device=DEVICE,
            round_number=1,
            client_data_source=client_data_source,
        )

        measured_round_time = (
            time.perf_counter()
            - round_start
        )

        fedavg_was_run = True

        # --------------------------------------------------------------
        # Validate round result
        # --------------------------------------------------------------

        print()
        print("FEDAVG ROUND COMPLETE")

        print(
            f"Round time: "
            f"{measured_round_time:.2f} seconds"
        )

        actual_client_count = len(
            round_result.client_results
        )

        actual_total_samples = (
            round_result.total_client_samples
        )

        print(
            f"Clients trained: "
            f"{actual_client_count}"
        )

        print(
            f"Samples aggregated: "
            f"{actual_total_samples:,}"
        )

        if round_result.round_number != 1:
            raise RuntimeError(
                "Unexpected federated round number."
            )

        if actual_client_count != NUM_CLIENTS:
            raise RuntimeError(
                "Unexpected number of trained clients. "
                f"Expected {NUM_CLIENTS}, "
                f"got {actual_client_count}."
            )

        if actual_total_samples != EXPECTED_TRAINING_ROWS:
            raise RuntimeError(
                "Unexpected aggregated sample count. "
                f"Expected {EXPECTED_TRAINING_ROWS:,}, "
                f"got {actual_total_samples:,}."
            )

        if not round_result.global_state_dict:
            raise RuntimeError(
                "FedAvg returned an empty global state dictionary."
            )

        print("Round number: PASS")
        print("Client count: PASS")
        print("Aggregated sample count: PASS")
        print("Global state dictionary: PASS")
        print()

        # --------------------------------------------------------------
        # Save global model immediately
        # --------------------------------------------------------------

        print("8. SAVING GLOBAL MODEL")

        torch.save(
            round_result.global_state_dict,
            MODEL_PATH,
        )

        if not MODEL_PATH.is_file():
            raise RuntimeError(
                "Global model file was not created."
            )

        print(
            f"Saved: {MODEL_PATH}"
        )

        print()

    # ------------------------------------------------------------------
    # 7. Load global model for validation
    # ------------------------------------------------------------------

    print("7. LOADING GLOBAL MODEL FOR VALIDATION")

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Global model does not exist:\n"
            f"{MODEL_PATH}"
        )

    validation_model = SmallMLP(
        input_dim=EXPECTED_FEATURE_COUNT
    )

    saved_state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    if not isinstance(
        saved_state_dict,
        dict,
    ):
        raise TypeError(
            "Saved global model does not contain "
            "a valid state dictionary."
        )

    validation_model.load_state_dict(
        saved_state_dict
    )

    validation_model.to(
        DEVICE
    )

    validation_model.eval()

    print("Saved model reload: PASS")
    print()

    # ------------------------------------------------------------------
    # 8. Validation
    # ------------------------------------------------------------------

    print("8. VALIDATING GLOBAL MODEL")

    validation_start = time.perf_counter()

    criterion = nn.BCELoss()

    validation_metrics = evaluate_batches(
        model=validation_model,
        batches=validation_tuple_batches(
            loader=loader,
            scaler=scaler,
            batch_size=BATCH_SIZE,
        ),
        criterion=criterion,
        device=DEVICE,
    )

    validation_time = (
        time.perf_counter()
        - validation_start
    )

    if validation_metrics.samples != EXPECTED_VALIDATION_ROWS:
        raise RuntimeError(
            "Validation row-count mismatch. "
            f"Expected {EXPECTED_VALIDATION_ROWS:,}, "
            f"got {validation_metrics.samples:,}."
        )

    print()
    print("VALIDATION COMPLETE")

    print(
        f"Samples:   "
        f"{validation_metrics.samples:,}"
    )

    print(
        f"Loss:      "
        f"{validation_metrics.loss:.12f}"
    )

    print(
        f"Accuracy:  "
        f"{validation_metrics.accuracy:.12f}"
    )

    print(
        f"Precision: "
        f"{validation_metrics.precision:.12f}"
    )

    print(
        f"Recall:    "
        f"{validation_metrics.recall:.12f}"
    )

    print(
        f"F1:        "
        f"{validation_metrics.f1:.12f}"
    )

    print(
        f"ROC-AUC:   "
        f"{validation_metrics.roc_auc:.12f}"
    )

    print(
        f"Validation time: "
        f"{validation_time:.2f} seconds"
    )

    print()

    # ------------------------------------------------------------------
    # 9. Save experiment result JSON
    # ------------------------------------------------------------------

    print("9. SAVING EXPERIMENT RESULT")

    if fedavg_was_run and round_result is not None:

        actual_client_count = len(
            round_result.client_results
        )

        actual_total_samples = (
            round_result.total_client_samples
        )

        aggregation_elapsed_seconds = (
            round_result.aggregation_elapsed_seconds
        )

        round_number = (
            round_result.round_number
        )

    else:

        # The training phase happened in the previous execution.
        # We know from the completed console output that it was:
        #
        #   round 1
        #   9 clients
        #   4,943,824 samples
        #   957.92 seconds wall time
        #
        # We do not invent an aggregation timing value here.
        # It is therefore recorded as unavailable for this resumed run.

        round_number = 1

        actual_client_count = NUM_CLIENTS

        actual_total_samples = EXPECTED_TRAINING_ROWS

        aggregation_elapsed_seconds = None

    result = {
        "experiment": (
            "device_non_iid_federated_pilot"
        ),
        "status": "completed",
        "seed": SEED,
        "partition_type": "device_non_iid",
        "clients": NUM_CLIENTS,
        "client_ids": CLIENT_IDS,
        "client_training_samples": (
            client_training_samples
        ),
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "device": DEVICE,
        "model_parameter_count": parameter_count,
        "fedavg_round": {
            "round_number": round_number,
            "client_count": actual_client_count,
            "total_samples": actual_total_samples,
            "aggregation_elapsed_seconds": (
                aggregation_elapsed_seconds
            ),
            "measured_wall_time_seconds": (
                measured_round_time
            ),
        },
        "global_model_path": str(
            MODEL_PATH
        ),
        "validation": {
            "samples": (
                validation_metrics.samples
            ),
            "loss": (
                validation_metrics.loss
            ),
            "accuracy": (
                validation_metrics.accuracy
            ),
            "precision": (
                validation_metrics.precision
            ),
            "recall": (
                validation_metrics.recall
            ),
            "f1": (
                validation_metrics.f1
            ),
            "roc_auc": (
                validation_metrics.roc_auc
            ),
            "evaluation_time_seconds": (
                validation_time
            ),
        },
        "notes": [
            (
                "Clients represent simulated logical "
                "IoT participants mapped to N-BaIoT devices."
            ),
            (
                "Device-based partitioning preserves "
                "the original device-specific training "
                "data distributions."
            ),
            (
                "No random redistribution of training "
                "rows was performed."
            ),
            (
                "The preprocessing scaler was fitted "
                "on centralized training data and reused "
                "by clients."
            ),
            (
                "This experiment does not implement "
                "differential privacy."
            ),
            (
                "This experiment does not implement "
                "secure aggregation."
            ),
            (
                "This is a one-round pilot and is not "
                "the final multi-round experiment."
            ),
            (
                "The validation stage was completed using "
                "the saved global model after FedAvg training."
            ),
        ],
    }

    RESULT_JSON_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not RESULT_JSON_PATH.is_file():
        raise RuntimeError(
            "Experiment result JSON was not created."
        )

    print(
        f"Saved result JSON: "
        f"{RESULT_JSON_PATH}"
    )

    print()

    # ------------------------------------------------------------------
    # Final status
    # ------------------------------------------------------------------

    print("=" * 72)
    print("DEVICE NON-IID FEDAVG PILOT: COMPLETE")
    print("=" * 72)


if __name__ == "__main__":
    main()