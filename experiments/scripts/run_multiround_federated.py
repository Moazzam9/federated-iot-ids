from __future__ import annotations

import argparse
import json
import time
from itertools import islice
from pathlib import Path
from typing import Iterator

import joblib
import torch
from torch import nn

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.data.torch_data import iter_torch_batches
from src.fl.client_data import (
    ClientDataSource,
    DeviceClientDataSource,
    IIDClientDataSource,
)
from src.fl.coordinator import run_federated_round
from src.fl.iid_data import IIDClientDataLoader
from src.models.mlp import SmallMLP
from src.models.trainer import EvaluationMetrics, evaluate_batches


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATASET_ROOT = Path(
    r"D:\federated-iot-temp"
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

RESULTS_ROOT = (
    PROJECT_ROOT
    / "results"
    / "raw"
)

MODEL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "raw"
)


# ----------------------------------------------------------------------
# Fixed experiment constants
# ----------------------------------------------------------------------

DEVICE_CLIENT_IDS = [
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

IID_CLIENT_IDS = [
    f"client_{index}"
    for index in range(1, 10)
]


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a reproducible multi-round FedAvg experiment "
            "on N-BaIoT."
        )
    )

    parser.add_argument(
        "--partition",
        choices=[
            "iid",
            "device_non_iid",
        ],
        required=True,
        help="Client partitioning strategy.",
    )

    parser.add_argument(
        "--rounds",
        type=int,
        default=1,
        help="Number of FedAvg rounds.",
    )

    parser.add_argument(
        "--local-epochs",
        type=int,
        default=1,
        help="Number of local epochs per client per round.",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=256,
        help="Training and validation batch size.",
    )

    parser.add_argument(
        "--learning-rate",
        type=float,
        default=0.001,
        help="Adam learning rate.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Experiment random seed.",
    )

    parser.add_argument(
        "--validation-max-batches",
        type=int,
        default=None,
        help=(
            "Optional maximum number of validation batches per round. "
            "Use a small value for smoke testing. "
            "Omit for full validation."
        ),
    )

    parser.add_argument(
        "--output-prefix",
        type=str,
        default=None,
        help=(
            "Optional output filename prefix. "
            "If omitted, one is generated from the partition."
        ),
    )

    return parser.parse_args()


def set_random_seed(seed: int) -> None:
    torch.manual_seed(seed)


def load_training_scaler() -> FittedStandardScaler:
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(
            "Training scaler does not exist:\n"
            f"{SCALER_PATH}"
        )

    scaler = joblib.load(SCALER_PATH)

    if not isinstance(
        scaler,
        FittedStandardScaler,
    ):
        raise TypeError(
            "Persisted scaler is not a FittedStandardScaler."
        )

    return scaler


def create_loader() -> NBaIoTSplitLoader:
    return NBaIoTSplitLoader(
        data_root=DATASET_ROOT,
        split_root=SPLIT_ROOT,
    )


def create_client_data_source(
    loader: NBaIoTSplitLoader,
    partition: str,
) -> tuple[list[str], ClientDataSource]:

    if partition == "device_non_iid":
        client_ids = list(DEVICE_CLIENT_IDS)

        data_source = DeviceClientDataSource(
            loader
        )

        return client_ids, data_source

    if partition == "iid":
        if not IID_ASSIGNMENT_PATH.is_file():
            raise FileNotFoundError(
                "IID assignment file does not exist:\n"
                f"{IID_ASSIGNMENT_PATH}"
            )

        iid_loader = IIDClientDataLoader(
            loader=loader,
            assignment_path=IID_ASSIGNMENT_PATH,
            num_clients=9,
        )

        data_source = IIDClientDataSource(
            iid_loader
        )

        return list(IID_CLIENT_IDS), data_source

    raise ValueError(
        f"Unsupported partition: {partition!r}"
    )


def validation_batches(
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
    batch_size: int,
    max_batches: int | None,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:

    batches = iter_torch_batches(
        loader=loader,
        scaler=scaler,
        split="validation",
        batch_size=batch_size,
        device="cpu",
    )

    if max_batches is not None:
        batches = islice(
            batches,
            max_batches,
        )

    for batch in batches:
        yield batch.features, batch.labels


def evaluate_global_model(
    model: nn.Module,
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
    batch_size: int,
    max_batches: int | None,
) -> tuple[EvaluationMetrics, float]:

    criterion = nn.BCELoss()

    start = time.perf_counter()

    metrics = evaluate_batches(
        model=model,
        batches=validation_batches(
            loader=loader,
            scaler=scaler,
            batch_size=batch_size,
            max_batches=max_batches,
        ),
        criterion=criterion,
        device="cpu",
    )

    elapsed = (
        time.perf_counter()
        - start
    )

    return metrics, float(elapsed)


def metrics_to_dict(
    metrics: EvaluationMetrics,
) -> dict[str, float | int]:

    return {
        "loss": float(metrics.loss),
        "samples": int(metrics.samples),
        "accuracy": float(metrics.accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "roc_auc": float(metrics.roc_auc),
    }


def save_model(
    model: nn.Module,
    path: Path,
) -> None:

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    torch.save(
        model.state_dict(),
        path,
    )


def validate_model_file(
    model: nn.Module,
    path: Path,
) -> None:

    if not path.is_file():
        raise FileNotFoundError(
            "Expected model checkpoint was not created:\n"
            f"{path}"
        )

    state_dict = torch.load(
        path,
        map_location="cpu",
        weights_only=True,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )


# ----------------------------------------------------------------------
# Main experiment
# ----------------------------------------------------------------------

def main() -> None:

    args = parse_args()

    if args.rounds <= 0:
        raise ValueError(
            "--rounds must be greater than zero."
        )

    if args.local_epochs <= 0:
        raise ValueError(
            "--local-epochs must be greater than zero."
        )

    if args.batch_size <= 0:
        raise ValueError(
            "--batch-size must be greater than zero."
        )

    if args.learning_rate <= 0:
        raise ValueError(
            "--learning-rate must be greater than zero."
        )

    if args.validation_max_batches is not None:
        if args.validation_max_batches <= 0:
            raise ValueError(
                "--validation-max-batches must be greater than zero."
            )

    set_random_seed(args.seed)

    print("=" * 72)
    print("MULTI-ROUND FEDAVG EXPERIMENT")
    print("=" * 72)

    print(
        f"Project root:       {PROJECT_ROOT}"
    )
    print(
        f"Dataset root:       {DATASET_ROOT}"
    )
    print(
        f"Split root:         {SPLIT_ROOT}"
    )
    print(
        f"Scaler path:        {SCALER_PATH}"
    )
    print(
        f"Partition:          {args.partition}"
    )
    print(
        f"Rounds:             {args.rounds}"
    )
    print(
        f"Local epochs:       {args.local_epochs}"
    )
    print(
        f"Batch size:         {args.batch_size}"
    )
    print(
        f"Learning rate:      {args.learning_rate}"
    )
    print(
        f"Seed:               {args.seed}"
    )

    if args.validation_max_batches is not None:
        print(
            "Validation batches: "
            f"{args.validation_max_batches}"
        )
    else:
        print(
            "Validation batches: ALL"
        )

    # --------------------------------------------------------------
    # Loader
    # --------------------------------------------------------------

    print()
    print("1. CREATING N-BAIoT LOADER")

    loader = create_loader()

    loader.validate_source_indexes()

    print("Source split indexes: PASS")

    training_rows = (
        loader.total_rows_for_split("train")
    )

    validation_rows = (
        loader.total_rows_for_split("validation")
    )

    print(
        f"Training rows:   "
        f"{training_rows:,}"
    )

    print(
        f"Validation rows: "
        f"{validation_rows:,}"
    )

    # --------------------------------------------------------------
    # Scaler
    # --------------------------------------------------------------

    print()
    print("2. LOADING TRAINING SCALER")

    scaler = load_training_scaler()

    print("Training-only scaler: PASS")

    # --------------------------------------------------------------
    # Clients
    # --------------------------------------------------------------

    print()
    print("3. CREATING CLIENT DATA SOURCE")

    client_ids, client_data_source = (
        create_client_data_source(
            loader=loader,
            partition=args.partition,
        )
    )

    print(
        f"Client count: {len(client_ids)}"
    )

    client_sample_counts: dict[str, int] = {}

    for client_id in client_ids:

        samples = (
            client_data_source.count_training_samples(
                client_id
            )
        )

        client_sample_counts[client_id] = int(
            samples
        )

        print(
            f"{client_id}: "
            f"{samples:,}"
        )

    total_client_samples = sum(
        client_sample_counts.values()
    )

    print(
        f"Total client samples: "
        f"{total_client_samples:,}"
    )

    if total_client_samples != training_rows:
        raise RuntimeError(
            "Client sample counts do not equal "
            "the frozen training split.\n"
            f"Client total: {total_client_samples:,}\n"
            f"Training rows: {training_rows:,}"
        )

    print("Client sample counts: PASS")

    # --------------------------------------------------------------
    # Global model
    # --------------------------------------------------------------

    print()
    print("4. CREATING GLOBAL MODEL")

    global_model = SmallMLP()

    parameter_count = sum(
        parameter.numel()
        for parameter in global_model.parameters()
        if parameter.requires_grad
    )

    print(
        f"Trainable parameters: "
        f"{parameter_count:,}"
    )

    # --------------------------------------------------------------
    # Output paths
    # --------------------------------------------------------------

    if args.output_prefix is None:
        output_prefix = (
            f"multiround_{args.partition}"
        )
    else:
        output_prefix = args.output_prefix

    result_path = (
        RESULTS_ROOT
        / f"{output_prefix}.json"
    )

    model_directory = (
        MODEL_ROOT
        / f"{output_prefix}_models"
    )

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    model_directory.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------------
    # Round loop
    # --------------------------------------------------------------

    round_results: list[dict] = []

    experiment_start = time.perf_counter()

    for round_number in range(
        1,
        args.rounds + 1,
    ):

        print()
        print("=" * 72)
        print(
            f"ROUND {round_number} / "
            f"{args.rounds}"
        )
        print("=" * 72)

        round_start = time.perf_counter()

        fedavg_result = run_federated_round(
            global_model=global_model,
            loader=loader,
            scaler=scaler,
            client_ids=client_ids,
            local_epochs=args.local_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device="cpu",
            round_number=round_number,
            client_data_source=client_data_source,
        )

        round_wall_time = (
            time.perf_counter()
            - round_start
        )

        print()
        print(
            "FedAvg training complete."
        )

        print(
            f"Round: "
            f"{fedavg_result.round_number}"
        )

        print(
            f"Clients: "
            f"{len(fedavg_result.client_results)}"
        )

        print(
            f"Samples: "
            f"{fedavg_result.total_client_samples:,}"
        )

        print(
            f"Aggregation time: "
            f"{fedavg_result.aggregation_elapsed_seconds:.6f} s"
        )

        print(
            f"Round wall time: "
            f"{round_wall_time:.2f} s"
        )

        # ----------------------------------------------------------
        # Save checkpoint immediately after FedAvg
        # ----------------------------------------------------------

        checkpoint_path = (
            model_directory
            / f"round_{round_number:03d}.pt"
        )

        save_model(
            global_model,
            checkpoint_path,
        )

        validate_model_file(
            SmallMLP(),
            checkpoint_path,
        )

        print(
            f"Checkpoint: {checkpoint_path}"
        )

        print("Checkpoint reload: PASS")

        # ----------------------------------------------------------
        # Validation
        # ----------------------------------------------------------

        print()
        print("Validation:")

        validation_metrics, validation_time = (
            evaluate_global_model(
                model=global_model,
                loader=loader,
                scaler=scaler,
                batch_size=args.batch_size,
                max_batches=args.validation_max_batches,
            )
        )

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
            f"{validation_time:.2f} s"
        )

        # ----------------------------------------------------------
        # Per-client training records
        # ----------------------------------------------------------

        client_round_results = []

        for client_result in (
            fedavg_result.client_results
        ):

            client_round_results.append(
                {
                    "client_id": client_result.client_id,
                    "samples": int(
                        client_result.samples
                    ),
                    "epochs": int(
                        client_result.epochs
                    ),
                    "loss": float(
                        client_result.loss
                    ),
                    "elapsed_seconds": float(
                        client_result.elapsed_seconds
                    ),
                }
            )

        round_results.append(
            {
                "round_number": round_number,
                "client_count": len(
                    fedavg_result.client_results
                ),
                "total_client_samples": int(
                    fedavg_result.total_client_samples
                ),
                "aggregation_elapsed_seconds": float(
                    fedavg_result.aggregation_elapsed_seconds
                ),
                "measured_wall_time_seconds": float(
                    round_wall_time
                ),
                "checkpoint_path": str(
                    checkpoint_path
                ),
                "validation": (
                    metrics_to_dict(
                        validation_metrics
                    )
                    | {
                        "evaluation_time_seconds":
                            validation_time
                    }
                ),
                "clients": client_round_results,
            }
        )

        # ----------------------------------------------------------
        # Save result after every completed round
        # ----------------------------------------------------------

        partial_result = {
            "experiment": output_prefix,
            "status": "in_progress",
            "partition_type": args.partition,
            "seed": args.seed,
            "clients": len(client_ids),
            "client_ids": client_ids,
            "local_epochs": args.local_epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": "cpu",
            "model_parameter_count": parameter_count,
            "validation_max_batches":
                args.validation_max_batches,
            "rounds_requested": args.rounds,
            "rounds_completed":
                len(round_results),
            "total_training_rows": training_rows,
            "total_validation_rows": validation_rows,
            "client_sample_counts":
                client_sample_counts,
            "rounds": round_results,
        }

        result_path.write_text(
            json.dumps(
                partial_result,
                indent=2,
            ),
            encoding="utf-8",
        )

    experiment_elapsed = (
        time.perf_counter()
        - experiment_start
    )

    # --------------------------------------------------------------
    # Final result
    # --------------------------------------------------------------

    final_result = {
        "experiment": output_prefix,
        "status": "completed",
        "partition_type": args.partition,
        "seed": args.seed,
        "clients": len(client_ids),
        "client_ids": client_ids,
        "local_epochs": args.local_epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "device": "cpu",
        "model_parameter_count": parameter_count,
        "validation_max_batches":
            args.validation_max_batches,
        "rounds_requested": args.rounds,
        "rounds_completed": len(round_results),
        "total_experiment_wall_time_seconds":
            float(experiment_elapsed),
        "total_training_rows": training_rows,
        "total_validation_rows": validation_rows,
        "client_sample_counts":
            client_sample_counts,
        "rounds": round_results,
        "notes": [
            (
                "Clients represent simulated logical IoT "
                "participants."
            ),
            (
                "The preprocessing scaler was fitted on "
                "centralized training data and reused by "
                "clients."
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
                "Validation uses the global validation split "
                "and does not modify the frozen training "
                "split."
            ),
        ],
    }

    result_path.write_text(
        json.dumps(
            final_result,
            indent=2,
        ),
        encoding="utf-8",
    )

    print()
    print("=" * 72)
    print("MULTI-ROUND FEDAVG EXPERIMENT: COMPLETE")
    print("=" * 72)

    print(
        f"Rounds completed: "
        f"{len(round_results)}"
    )

    print(
        f"Total experiment time: "
        f"{experiment_elapsed:.2f} s"
    )

    print(
        f"Result JSON: "
        f"{result_path}"
    )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    main()