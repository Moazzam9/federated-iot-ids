from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterator

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn


@dataclass(frozen=True)
class TrainingMetrics:
    """Metrics produced after one training epoch."""

    loss: float
    samples: int
    elapsed_seconds: float


@dataclass(frozen=True)
class EvaluationMetrics:
    """Binary-classification evaluation metrics."""

    loss: float
    samples: int
    accuracy: float
    precision: float
    recall: float
    f1: float
    roc_auc: float


def set_random_seed(seed: int) -> None:
    """Set random seeds for reproducible CPU-based experiments."""
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module,
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> TrainingMetrics:
    """Train a model for one epoch over streamed batches."""
    if device != "cpu":
        raise ValueError(
            "Only CPU execution is currently supported."
        )

    model.train()

    total_loss = 0.0
    total_samples = 0

    start_time = perf_counter()

    for features, labels in batches:
        features = features.to(device)
        labels = labels.to(device)

        optimizer.zero_grad(set_to_none=True)

        predictions = model(features)
        loss = criterion(predictions, labels)

        loss.backward()
        optimizer.step()

        batch_size = features.shape[0]

        total_loss += loss.item() * batch_size
        total_samples += batch_size

    elapsed_seconds = perf_counter() - start_time

    if total_samples == 0:
        raise ValueError(
            "No samples were provided to train_one_epoch()."
        )

    return TrainingMetrics(
        loss=total_loss / total_samples,
        samples=total_samples,
        elapsed_seconds=elapsed_seconds,
    )


def evaluate_batches(
    model: nn.Module,
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: str = "cpu",
) -> EvaluationMetrics:
    """
    Evaluate a binary classifier over streamed batches.

    Labels and predicted probabilities are accumulated on the CPU
    so the complete feature matrix does not need to remain in memory.
    """
    if device != "cpu":
        raise ValueError(
            "Only CPU execution is currently supported."
        )

    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    with torch.no_grad():
        for features, labels in batches:
            features = features.to(device)
            labels = labels.to(device)

            probabilities = model(features)
            loss = criterion(probabilities, labels)

            batch_size = features.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size

            all_labels.append(
                labels.detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

            all_probabilities.append(
                probabilities.detach()
                .cpu()
                .numpy()
                .reshape(-1)
            )

    if total_samples == 0:
        raise ValueError(
            "No samples were provided to evaluate_batches()."
        )

    labels_array = np.concatenate(all_labels).astype(np.int64)
    probabilities_array = np.concatenate(all_probabilities)

    predictions_array = (
        probabilities_array >= 0.5
    ).astype(np.int64)

    accuracy = accuracy_score(
        labels_array,
        predictions_array,
    )

    precision = precision_score(
        labels_array,
        predictions_array,
        zero_division=0,
    )

    recall = recall_score(
        labels_array,
        predictions_array,
        zero_division=0,
    )

    f1 = f1_score(
        labels_array,
        predictions_array,
        zero_division=0,
    )

    if np.unique(labels_array).size < 2:
        roc_auc = 0.5
    else:
        roc_auc = roc_auc_score(
            labels_array,
            probabilities_array,
        )

    return EvaluationMetrics(
        loss=total_loss / total_samples,
        samples=total_samples,
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        roc_auc=float(roc_auc),
    )
