from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from time import perf_counter

import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.torch_data import iter_torch_batches
from src.models.mlp import SmallMLP
from src.models.trainer import (
    evaluate_batches,
    set_random_seed,
    train_one_epoch,
)


CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "training_standard_scaler.pkl"
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "raw" / "centralized"
MODEL_ROOT = RESULTS_ROOT / "models"


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

    if (
        config["project"]["random_seed"]
        != config["reproducibility"]["seed"]
    ):
        raise ValueError(
            "project.random_seed and reproducibility.seed must match."
        )

    if config["dataset"]["expected_features"] != 115:
        raise ValueError(
            "Dataset expected_features must be 115."
        )

    architecture = config["model"]["architecture"]

    if architecture["input_dim"] != 115:
        raise ValueError("Model input_dim must be 115.")

    if architecture["hidden_layers"] != [64, 32]:
        raise ValueError(
            "Model hidden_layers must be [64, 32]."
        )

    if architecture["output_dim"] != 1:
        raise ValueError("Model output_dim must be 1.")

    if config["training"]["batch_size"] <= 0:
        raise ValueError(
            "Training batch_size must be greater than zero."
        )

    if config["training"]["max_centralized_epochs"] <= 0:
        raise ValueError(
            "max_centralized_epochs must be greater than zero."
        )


def load_training_scaler():
    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            "Training scaler was not found:\n"
            f"{SCALER_PATH}"
        )

    with SCALER_PATH.open("rb") as handle:
        return pickle.load(handle)


def build_model(config: dict) -> SmallMLP:
    architecture = config["model"]["architecture"]

    return SmallMLP(
        input_dim=architecture["input_dim"],
        hidden_dim_1=architecture["hidden_layers"][0],
        hidden_dim_2=architecture["hidden_layers"][1],
    )


def make_batch_iterator(
    loader: NBaIoTSplitLoader,
    scaler,
    split: str,
    batch_size: int,
    max_rows: int | None = None,
):
    if max_rows is not None and max_rows <= 0:
        raise ValueError("max_rows must be greater than zero.")

    rows_seen = 0

    for batch in iter_torch_batches(
        loader=loader,
        scaler=scaler,
        split=split,
        batch_size=batch_size,
        device="cpu",
    ):
        features = batch.features
        labels = batch.labels

        if max_rows is not None:
            remaining = max_rows - rows_seen

            if remaining <= 0:
                break

            if features.shape[0] > remaining:
                features = features[:remaining]
                labels = labels[:remaining]

        rows_in_batch = features.shape[0]

        if rows_in_batch == 0:
            break

        yield features, labels

        rows_seen += rows_in_batch

        if max_rows is not None and rows_seen >= max_rows:
            break


def count_trainable_parameters(model: torch.nn.Module) -> int:
    return sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the centralized N-BaIoT baseline."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of centralized training epochs.",
    )

    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=None,
        help=(
            "Optional maximum number of training rows "
            "for a controlled pilot."
        ),
    )

    parser.add_argument(
        "--max-validation-rows",
        type=int,
        default=None,
        help=(
            "Optional maximum number of validation rows "
            "for a controlled pilot."
        ),
    )

    args = parser.parse_args()

    config = load_config()
    validate_config(config)

    seed = config["reproducibility"]["seed"]
    batch_size = config["training"]["batch_size"]
    configured_epochs = config["training"]["max_centralized_epochs"]

    epochs = (
        args.epochs
        if args.epochs is not None
        else configured_epochs
    )

    if epochs <= 0:
        raise ValueError("--epochs must be greater than zero.")

    set_random_seed(seed)

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    MODEL_ROOT.mkdir(parents=True, exist_ok=True)

    scaler = load_training_scaler()

    loader = NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        chunk_size=50_000,
    )

    print("CENTRALIZED N-BaIoT BASELINE")
    print(f"Seed:                  {seed}")
    print(f"Epochs:                {epochs}")
    print(f"Batch size:            {batch_size}")
    print(f"Max train rows:        {args.max_train_rows}")
    print(f"Max validation rows:   {args.max_validation_rows}")
    print("Model:                 SmallMLP")
    print("Input features:        115")
    print("Hidden layers:         [64, 32]")
    print("Device:                CPU")
    print()

    print("Validating source split indexes...")
    loader.validate_source_indexes()
    print("Source split indexes: PASS")
    print()

    model = build_model(config)
    criterion = torch.nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config["model"]["learning_rate"],
    )

    parameter_count = count_trainable_parameters(model)

    print(f"Trainable parameters: {parameter_count:,}")
    print()

    history = []

    best_f1 = -1.0
    best_epoch = None

    experiment_start = perf_counter()

    for epoch in range(1, epochs + 1):
        epoch_start = perf_counter()

        print(f"--- Epoch {epoch}/{epochs} ---")

        train_batches = make_batch_iterator(
            loader=loader,
            scaler=scaler,
            split="train",
            batch_size=batch_size,
            max_rows=args.max_train_rows,
        )

        train_metrics = train_one_epoch(
            model=model,
            batches=train_batches,
            optimizer=optimizer,
            criterion=criterion,
            device="cpu",
        )

        validation_batches = make_batch_iterator(
            loader=loader,
            scaler=scaler,
            split="validation",
            batch_size=batch_size,
            max_rows=args.max_validation_rows,
        )

        validation_metrics = evaluate_batches(
            model=model,
            batches=validation_batches,
            criterion=criterion,
            device="cpu",
        )

        epoch_elapsed = perf_counter() - epoch_start

        print(
            f"Train loss: {train_metrics.loss:.6f} | "
            f"Train samples: {train_metrics.samples:,}"
        )

        print(
            f"Validation loss: {validation_metrics.loss:.6f} | "
            f"Validation samples: {validation_metrics.samples:,}"
        )

        print(
            f"Validation accuracy: "
            f"{validation_metrics.accuracy:.6f} | "
            f"precision: {validation_metrics.precision:.6f} | "
            f"recall: {validation_metrics.recall:.6f} | "
            f"F1: {validation_metrics.f1:.6f} | "
            f"ROC-AUC: {validation_metrics.roc_auc:.6f}"
        )

        print(
            f"Epoch elapsed: {epoch_elapsed:.2f} seconds"
        )

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_metrics.loss,
                "train_samples": train_metrics.samples,
                "train_elapsed_seconds": (
                    train_metrics.elapsed_seconds
                ),
                "validation_loss": validation_metrics.loss,
                "validation_samples": validation_metrics.samples,
                "validation_accuracy": (
                    validation_metrics.accuracy
                ),
                "validation_precision": (
                    validation_metrics.precision
                ),
                "validation_recall": validation_metrics.recall,
                "validation_f1": validation_metrics.f1,
                "validation_roc_auc": (
                    validation_metrics.roc_auc
                ),
                "epoch_elapsed_seconds": epoch_elapsed,
            }
        )

        if validation_metrics.f1 > best_f1:
            best_f1 = validation_metrics.f1
            best_epoch = epoch

            best_model_path = (
                MODEL_ROOT / "best_centralized_model.pt"
            )

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "validation_f1": validation_metrics.f1,
                    "parameter_count": parameter_count,
                    "seed": seed,
                },
                best_model_path,
            )

            print(
                f"New best model saved: {best_model_path}"
            )

        print()

    total_elapsed = perf_counter() - experiment_start

    history_path = (
        RESULTS_ROOT / "centralized_training_history.json"
    )

    result = {
        "experiment": "centralized",
        "dataset": config["dataset"]["name"],
        "model": config["model"]["name"],
        "seed": seed,
        "epochs": epochs,
        "batch_size": batch_size,
        "parameter_count": parameter_count,
        "max_train_rows": args.max_train_rows,
        "max_validation_rows": args.max_validation_rows,
        "best_epoch": best_epoch,
        "best_validation_f1": best_f1,
        "total_elapsed_seconds": total_elapsed,
        "history": history,
    }

    with history_path.open("w", encoding="utf-8") as handle:
        json.dump(
            result,
            handle,
            indent=2,
        )

    print("CENTRALIZED EXPERIMENT COMPLETED")
    print(f"Best epoch:             {best_epoch}")
    print(f"Best validation F1:     {best_f1:.6f}")
    print(f"Total elapsed seconds:  {total_elapsed:.2f}")
    print(f"Results:                {history_path}")
    print(
        "Best model:             "
        f"{MODEL_ROOT / 'best_centralized_model.pt'}"
    )


if __name__ == "__main__":
    main()