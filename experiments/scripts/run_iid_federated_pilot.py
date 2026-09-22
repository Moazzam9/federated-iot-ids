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
from src.fl.client_data import IIDClientDataSource
from src.fl.coordinator import run_federated_round
from src.fl.iid_data import IIDClientDataLoader
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

IID_ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fl"
    / "iid"
    / "train_client_assignments.npy"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "iid_federated_pilot.json"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "iid_federated_pilot_global_model.pt"
)

BATCH_SIZE = 256
LOCAL_EPOCHS = 1
LEARNING_RATE = 0.001
NUM_CLIENTS = 9
DEVICE = "cpu"


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
    """
    Adapt BatchData objects produced by iter_torch_batches()
    to the tuple interface expected by evaluate_batches().
    """

    batches = iter_torch_batches(
        loader=loader,
        scaler=scaler,
        split="validation",
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    for batch in batches:
        yield batch.features, batch.labels


def evaluate_global_model(
    model: SmallMLP,
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
):
    criterion = nn.BCELoss()

    return evaluate_batches(
        model=model,
        batches=validation_tuple_batches(
            loader=loader,
            scaler=scaler,
        ),
        criterion=criterion,
        device=DEVICE,
    )


def main() -> None:
    start_time = time.perf_counter()

    torch.set_num_threads(1)
    torch.manual_seed(42)

    loader = NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        chunk_size=50_000,
    )

    loader.validate_source_indexes()

    scaler = load_scaler()

    iid_loader = IIDClientDataLoader(
        loader=loader,
        assignment_path=IID_ASSIGNMENT_PATH,
        num_clients=NUM_CLIENTS,
    )

    client_data_source = IIDClientDataSource(
        iid_loader
    )

    client_ids = [
        f"client_{client_number}"
        for client_number in range(1, NUM_CLIENTS + 1)
    ]

    global_model = SmallMLP()

    print("Starting IID federated pilot.")
    print(f"Clients: {NUM_CLIENTS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Device: {DEVICE}")
    print()

    client_counts = {
        client_id: client_data_source.count_training_samples(
            client_id
        )
        for client_id in client_ids
    }

    print("Client training sample counts:")
    for client_id, count in client_counts.items():
        print(f"  {client_id}: {count:,}")

    print()
    print("Running FedAvg round 1...")

    round_start = time.perf_counter()

    round_result = run_federated_round(
        global_model=global_model,
        loader=loader,
        scaler=scaler,
        client_ids=client_ids,
        local_epochs=LOCAL_EPOCHS,
        batch_size=BATCH_SIZE,
        learning_rate=LEARNING_RATE,
        device=DEVICE,
        round_number=1,
        client_data_source=client_data_source,
    )

    round_seconds = time.perf_counter() - round_start

    print()
    print("FedAvg round 1 completed.")
    print(f"Round time: {round_seconds:.2f} seconds")

    # Save the trained global model immediately after FedAvg.
    MODEL_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        global_model.state_dict(),
        MODEL_PATH,
    )

    print(f"Saved global model: {MODEL_PATH}")
    print()
    print("Evaluating global model on validation split...")

    validation_start = time.perf_counter()

    validation_metrics = evaluate_global_model(
        model=global_model,
        loader=loader,
        scaler=scaler,
    )

    validation_seconds = (
        time.perf_counter() - validation_start
    )

    elapsed_seconds = time.perf_counter() - start_time

    parameter_count = sum(
        parameter.numel()
        for parameter in global_model.parameters()
    )

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
            "round_number": round_result.round_number,
            "client_count": len(round_result.client_results),
            "total_samples": round_result.total_samples,
            "training_time_seconds": round_seconds,
        },
        "model_parameter_count": parameter_count,
        "global_model_path": str(MODEL_PATH),
        "validation": {
            "samples": validation_metrics.samples,
            "loss": validation_metrics.loss,
            "accuracy": validation_metrics.accuracy,
            "precision": validation_metrics.precision,
            "recall": validation_metrics.recall,
            "f1": validation_metrics.f1,
            "roc_auc": validation_metrics.roc_auc,
            "evaluation_time_seconds": validation_seconds,
        },
        "elapsed_seconds": elapsed_seconds,
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
    print("IID federated pilot completed.")
    print(
        f"Validation accuracy: "
        f"{validation_metrics.accuracy:.6f}"
    )
    print(
        f"Validation F1: "
        f"{validation_metrics.f1:.6f}"
    )
    print(
        f"Validation ROC-AUC: "
        f"{validation_metrics.roc_auc:.6f}"
    )
    print(
        f"Round time: "
        f"{round_seconds:.2f} seconds"
    )
    print(
        f"Validation time: "
        f"{validation_seconds:.2f} seconds"
    )
    print(
        f"Total time: "
        f"{elapsed_seconds:.2f} seconds"
    )
    print()
    print(f"Saved result: {RESULTS_PATH}")
    print(f"Saved model: {MODEL_PATH}")


if __name__ == "__main__":
    main()