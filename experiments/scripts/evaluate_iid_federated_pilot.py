from __future__ import annotations

import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import joblib
import torch
import torch.nn as nn

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.data.torch_data import iter_torch_batches
from src.models.mlp import SmallMLP
from src.models.trainer import evaluate_batches


DATA_ROOT = (
    PROJECT_ROOT.parent
    / "federated-iot-temp"
    / "nbaiot_duplicate_8f_7x_ma"
)

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

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "iid_federated_pilot_global_model.pt"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "iid_federated_pilot.json"
)

BATCH_SIZE = 256
DEVICE = "cpu"

# Measured during the completed FedAvg training run.
FEDAVG_ROUND_SECONDS = 5000.85

NUM_CLIENTS = 9
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.001


def load_scaler() -> FittedStandardScaler:
    scaler = joblib.load(SCALER_PATH)

    if not isinstance(scaler, FittedStandardScaler):
        raise TypeError(
            "Persisted scaler is not a FittedStandardScaler."
        )

    return scaler


def validation_tuple_batches(
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
):
    batches = iter_torch_batches(
        loader=loader,
        scaler=scaler,
        split="validation",
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    for batch in batches:
        yield batch.features, batch.labels


def main() -> None:
    print("Loading saved IID FedAvg global model...")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Saved model not found: {MODEL_PATH}"
        )

    model = SmallMLP()

    state_dict = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print(f"Loaded model: {MODEL_PATH}")

    loader = NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        chunk_size=50_000,
    )

    loader.validate_source_indexes()

    scaler = load_scaler()
    criterion = nn.BCELoss()

    print()
    print("Evaluating IID FedAvg model on validation split...")
    print()

    validation_start = time.perf_counter()

    metrics = evaluate_batches(
        model=model,
        batches=validation_tuple_batches(
            loader=loader,
            scaler=scaler,
        ),
        criterion=criterion,
        device=DEVICE,
    )

    validation_seconds = (
        time.perf_counter() - validation_start
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
    )

    client_ids = [
        f"client_{client_number}"
        for client_number in range(1, NUM_CLIENTS + 1)
    ]

    client_counts = {
        "client_1": 549315,
        "client_2": 549315,
        "client_3": 549314,
        "client_4": 549314,
        "client_5": 549314,
        "client_6": 549313,
        "client_7": 549313,
        "client_8": 549313,
        "client_9": 549313,
    }

    result = {
        "experiment": "iid_federated_pilot",
        "status": "completed",
        "seed": 42,
        "clients": NUM_CLIENTS,
        "client_ids": client_ids,
        "client_training_samples": client_counts,
        "local_epochs": LOCAL_EPOCHS,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "device": DEVICE,
        "fedavg_round": {
            "round_number": 1,
            "client_count": NUM_CLIENTS,
            "total_samples": 4_943_824,
            "training_time_seconds": FEDAVG_ROUND_SECONDS,
        },
        "model_parameter_count": parameter_count,
        "global_model_path": str(MODEL_PATH),
        "validation": {
            "samples": metrics.samples,
            "loss": metrics.loss,
            "accuracy": metrics.accuracy,
            "precision": metrics.precision,
            "recall": metrics.recall,
            "f1": metrics.f1,
            "roc_auc": metrics.roc_auc,
            "evaluation_time_seconds": validation_seconds,
        },
        "note": (
            "Validation metrics were obtained from the saved global "
            "model after the completed IID FedAvg pilot. The model "
            "was not retrained during this validation-only run."
        ),
    }

    RESULTS_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("IID FedAvg validation completed.")
    print()
    print(f"Samples:   {metrics.samples:,}")
    print(f"Loss:      {metrics.loss:.6f}")
    print(f"Accuracy:  {metrics.accuracy:.6f}")
    print(f"Precision: {metrics.precision:.6f}")
    print(f"Recall:    {metrics.recall:.6f}")
    print(f"F1:        {metrics.f1:.6f}")
    print(f"ROC-AUC:   {metrics.roc_auc:.6f}")
    print(f"Validation time: {validation_seconds:.2f} seconds")
    print()
    print(f"Saved result: {RESULTS_PATH}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()