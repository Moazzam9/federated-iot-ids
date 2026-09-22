from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import yaml

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.data.torch_data import iter_torch_batches
from src.fl.client import train_client
from src.fl.client_data import DeviceClientDataSource
from src.models.mlp import SmallMLP
from src.models.trainer import evaluate_batches, set_random_seed


PROJECT_ROOT = Path(__file__).resolve().parents[2]

CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "training_standard_scaler.pkl"
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "raw" / "local_only"
MODEL_ROOT = RESULTS_ROOT / "models"

DATASET_ROOT = Path(r"D:\federated-iot-temp")

SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)


DEVICE_IDS = [
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


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def validate_config(config: dict) -> None:
    required_sections = [
        "project",
        "dataset",
        "preprocessing",
        "model",
        "training",
        "evaluation",
        "reproducibility",
    ]

    for section in required_sections:
        if section not in config:
            raise ValueError(
                f"Missing required configuration section: {section}"
            )

    if config["project"]["random_seed"] != config["reproducibility"]["seed"]:
        raise ValueError(
            "project.random_seed and reproducibility.seed must match."
        )

    if config["dataset"]["expected_features"] != 115:
        raise ValueError(
            "Dataset expected_features must be 115."
        )

    if config["dataset"]["expected_devices"] != len(DEVICE_IDS):
        raise ValueError(
            "Configured expected device count does not match DEVICE_IDS."
        )

    architecture = config["model"]["architecture"]

    if architecture["input_dim"] != 115:
        raise ValueError(
            "Model input_dim must be 115."
        )

    if architecture["hidden_layers"] != [64, 32]:
        raise ValueError(
            "Model hidden_layers must be [64, 32]."
        )

    if architecture["output_dim"] != 1:
        raise ValueError(
            "Model output_dim must be 1."
        )

    if config["training"]["batch_size"] <= 0:
        raise ValueError(
            "Training batch_size must be greater than zero."
        )

    if config["training"]["max_local_only_epochs"] <= 0:
        raise ValueError(
            "max_local_only_epochs must be greater than zero."
        )


def load_training_scaler() -> FittedStandardScaler:
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Training scaler was not found:\n"
            f"{SCALER_PATH}"
        )

    import joblib

    scaler = joblib.load(SCALER_PATH)

    if not isinstance(scaler, FittedStandardScaler):
        raise TypeError(
            "Persisted training scaler is not a "
            "FittedStandardScaler."
        )

    return scaler


def build_model(config: dict) -> SmallMLP:
    architecture = config["model"]["architecture"]

    return SmallMLP(
        input_dim=architecture["input_dim"],
        hidden_dim_1=architecture["hidden_layers"][0],
        hidden_dim_2=architecture["hidden_layers"][1],
    )


def make_validation_batches(
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
    device_id: str,
    batch_size: int,
):
    for batch in iter_torch_batches(
        loader=loader,
        scaler=scaler,
        split="validation",
        devices=[device_id],
        batch_size=batch_size,
        device="cpu",
    ):
        yield batch.features, batch.labels


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def calculate_macro_average(
    client_results: list[dict],
    metric_name: str,
) -> float:
    values = [
        float(result[metric_name])
        for result in client_results
    ]

    if not values:
        raise ValueError(
            "Cannot calculate macro average from no client results."
        )

    return sum(values) / len(values)


def calculate_weighted_average(
    client_results: list[dict],
    metric_name: str,
) -> float:
    total_samples = sum(
        int(result["validation_samples"])
        for result in client_results
    )

    if total_samples <= 0:
        raise ValueError(
            "Cannot calculate weighted average with zero samples."
        )

    weighted_sum = sum(
        float(result[metric_name])
        * int(result["validation_samples"])
        for result in client_results
    )

    return weighted_sum / total_samples


def save_model(
    model: torch.nn.Module,
    path: Path,
    device_id: str,
    epochs: int,
    seed: int,
    parameter_count: int,
) -> None:
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "device_id": device_id,
            "epochs": epochs,
            "seed": seed,
            "parameter_count": parameter_count,
        },
        path,
    )


def validate_saved_model(
    model_path: Path,
    config: dict,
) -> None:
    checkpoint = torch.load(
        model_path,
        map_location="cpu",
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            f"Missing model_state_dict in {model_path}"
        )

    model = build_model(config)

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the N-BaIoT Local-only baseline with one "
            "independent model per device."
        )
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help=(
            "Number of local training epochs. "
            "Defaults to training.max_local_only_epochs."
        ),
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help=(
            "Random seed. Defaults to "
            "reproducibility.seed from the config."
        ),
    )

    args = parser.parse_args()

    config = load_config()
    validate_config(config)

    configured_seed = config["reproducibility"]["seed"]

    seed = (
        args.seed
        if args.seed is not None
        else configured_seed
    )

    configured_epochs = config["training"]["max_local_only_epochs"]

    epochs = (
        args.epochs
        if args.epochs is not None
        else configured_epochs
    )

    if epochs <= 0:
        raise ValueError(
            "--epochs must be greater than zero."
        )

    batch_size = config["training"]["batch_size"]
    learning_rate = config["model"]["learning_rate"]

    set_random_seed(seed)

    RESULTS_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("LOCAL-ONLY N-BaIoT BASELINE")
    print(f"Seed:                  {seed}")
    print(f"Epochs per client:     {epochs}")
    print(f"Batch size:            {batch_size}")
    print(f"Learning rate:         {learning_rate}")
    print(f"Clients/devices:       {len(DEVICE_IDS)}")
    print("Model:                 SmallMLP")
    print("Input features:        115")
    print("Hidden layers:         [64, 32]")
    print("Device:                CPU")
    print()

    print("Loading training scaler...")
    scaler = load_training_scaler()
    print("Training scaler: PASS")
    print()

    print("Creating N-BaIoT loader...")
    loader = NBaIoTSplitLoader(
        data_root=DATASET_ROOT,
        split_root=SPLIT_ROOT,
        chunk_size=50_000,
    )

    print("Validating source split indexes...")
    loader.validate_source_indexes()
    print("Source split indexes: PASS")
    print()

    print("Creating device client data source...")
    data_source = DeviceClientDataSource(loader)

    total_training_rows = loader.total_rows_for_split("train")
    total_validation_rows = loader.total_rows_for_split(
        "validation"
    )

    print(
        f"Total training rows:  {total_training_rows:,}"
    )
    print(
        f"Total validation rows:{total_validation_rows:,}"
    )
    print()

    client_sample_counts = {}

    for device_id in DEVICE_IDS:
        client_sample_counts[device_id] = (
            data_source.count_training_samples(device_id)
        )

    counted_training_rows = sum(
        client_sample_counts.values()
    )

    if counted_training_rows != total_training_rows:
        raise RuntimeError(
            "Device client training sample counts do not "
            "sum to the frozen global training row count.\n"
            f"Client total: {counted_training_rows:,}\n"
            f"Global total: {total_training_rows:,}"
        )

    print(
        "Device client training counts: PASS"
    )

    for device_id in DEVICE_IDS:
        print(
            f"  {device_id}: "
            f"{client_sample_counts[device_id]:,}"
        )

    print()

    parameter_count = count_trainable_parameters(
        build_model(config)
    )

    print(
        f"Trainable parameters: {parameter_count:,}"
    )
    print()

    criterion = torch.nn.BCELoss()

    client_results = []

    experiment_start = time.perf_counter()

    for client_number, device_id in enumerate(
        DEVICE_IDS,
        start=1,
    ):
        print(
            f"--- Client {client_number}/{len(DEVICE_IDS)} ---"
        )
        print(f"Device: {device_id}")
        print(
            f"Training samples: "
            f"{client_sample_counts[device_id]:,}"
        )

        set_random_seed(seed)

        model = build_model(config)

        client_start = time.perf_counter()

        training_result = train_client(
            model=model,
            loader=loader,
            scaler=scaler,
            client_id=device_id,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device="cpu",
            client_data_source=data_source,
        )

        training_elapsed = (
            time.perf_counter() - client_start
        )

        validation_start = time.perf_counter()

        validation_metrics = evaluate_batches(
            model=model,
            batches=make_validation_batches(
                loader=loader,
                scaler=scaler,
                device_id=device_id,
                batch_size=batch_size,
            ),
            criterion=criterion,
            device="cpu",
        )

        validation_elapsed = (
            time.perf_counter() - validation_start
        )

        model_path = (
            MODEL_ROOT
            / f"{device_id}.pt"
        )

        save_model(
            model=model,
            path=model_path,
            device_id=device_id,
            epochs=epochs,
            seed=seed,
            parameter_count=parameter_count,
        )

        validate_saved_model(
            model_path=model_path,
            config=config,
        )

        print(
            f"Training loss:        "
            f"{training_result.loss:.6f}"
        )
        print(
            f"Validation loss:      "
            f"{validation_metrics.loss:.6f}"
        )
        print(
            f"Validation samples:   "
            f"{validation_metrics.samples:,}"
        )
        print(
            f"Validation accuracy:  "
            f"{validation_metrics.accuracy:.6f}"
        )
        print(
            f"Validation precision: "
            f"{validation_metrics.precision:.6f}"
        )
        print(
            f"Validation recall:    "
            f"{validation_metrics.recall:.6f}"
        )
        print(
            f"Validation F1:        "
            f"{validation_metrics.f1:.6f}"
        )
        print(
            f"Validation ROC-AUC:   "
            f"{validation_metrics.roc_auc:.6f}"
        )
        print(
            f"Training elapsed:     "
            f"{training_elapsed:.2f} seconds"
        )
        print(
            f"Validation elapsed:  "
            f"{validation_elapsed:.2f} seconds"
        )
        print(
            f"Model checkpoint:     {model_path}"
        )
        print()

        client_results.append(
            {
                "client_id": device_id,
                "training_samples": (
                    training_result.samples
                ),
                "validation_samples": (
                    validation_metrics.samples
                ),
                "epochs": training_result.epochs,
                "training_loss": (
                    training_result.loss
                ),
                "training_elapsed_seconds": (
                    training_elapsed
                ),
                "validation_elapsed_seconds": (
                    validation_elapsed
                ),
                "validation_loss": (
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
                "model_path": str(
                    model_path
                ),
            }
        )

    total_elapsed = (
        time.perf_counter() - experiment_start
    )

    metrics = [
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    macro_metrics = {
        metric: calculate_macro_average(
            client_results,
            metric,
        )
        for metric in metrics
    }

    weighted_metrics = {
        metric: calculate_weighted_average(
            client_results,
            metric,
        )
        for metric in metrics
    }

    result = {
        "experiment": "local_only",
        "status": "completed",
        "dataset": config["dataset"]["name"],
        "model": config["model"]["name"],
        "partition_type": "device_non_iid",
        "seed": seed,
        "client_count": len(DEVICE_IDS),
        "client_ids": DEVICE_IDS,
        "epochs_per_client": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "device": "cpu",
        "model_parameter_count": parameter_count,
        "total_training_rows": total_training_rows,
        "total_validation_rows": total_validation_rows,
        "client_training_sample_counts": (
            client_sample_counts
        ),
        "clients": client_results,
        "macro_average": macro_metrics,
        "sample_weighted_average": weighted_metrics,
        "total_experiment_wall_time_seconds": (
            total_elapsed
        ),
        "notes": [
            (
                "Each N-BaIoT device is treated as one "
                "simulated logical IoT client."
            ),
            (
                "Each local-only client trains an independent "
                "model using only its own device training data."
            ),
            (
                "No model parameters are exchanged between "
                "clients."
            ),
            (
                "Each local model is evaluated only on the "
                "validation data belonging to the same device."
            ),
            (
                "Macro averages give each client equal weight; "
                "sample-weighted averages weight clients by "
                "their validation sample counts."
            ),
            (
                "The preprocessing scaler was fitted on the "
                "centralized training data and reused by all "
                "clients."
            ),
            (
                "No differential privacy or secure aggregation "
                "is used."
            ),
            (
                "The frozen train/validation split indexes are "
                "not modified."
            ),
        ],
    }

    result_path = (
        RESULTS_ROOT
        / "local_only_results.json"
    )

    with result_path.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    print("LOCAL-ONLY EXPERIMENT COMPLETED")
    print(
        f"Total elapsed seconds: "
        f"{total_elapsed:.2f}"
    )
    print()
    print("Macro-average metrics:")

    for metric, value in macro_metrics.items():
        print(
            f"  {metric}: {value:.6f}"
        )

    print()
    print("Sample-weighted metrics:")

    for metric, value in weighted_metrics.items():
        print(
            f"  {metric}: {value:.6f}"
        )

    print()
    print(
        f"Results: {result_path}"
    )


if __name__ == "__main__":
    main()