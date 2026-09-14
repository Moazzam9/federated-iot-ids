from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterator, Optional

import numpy as np
import torch
from torch import nn

from src.models.mlp import SmallMLP


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
    """
    Set random seeds for reproducible CPU-based experiments.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: SmallMLP,
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str = "cpu",
) -> TrainingMetrics:
    """
    Train a model for one epoch over a streamed sequence of batches.
    """
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
    model: SmallMLP,
    batches: Iterator[tuple[torch.Tensor, torch.Tensor]],
    criterion: nn.Module,
    device: str = "cpu",
) -> EvaluationMetrics:
    """
    Evaluate a binary classifier over streamed batches.

    Predictions are accumulated as CPU NumPy arrays so that the
    complete validation/test feature matrix does not need to remain
    in memory.
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

    labels_array = np.concatenate(all_labels)
    probabilities_array = np.concatenate(all_probabilities)

    predictions_array = (
        probabilities_array >= 0.5
    ).astype(np.int64)

    labels_int = labels_array.astype(np.int64)

    true_positive = np.sum(
        (predictions_array == 1)
        & (labels_int == 1)
    )

    true_negative = np.sum(
        (predictions_array == 0)
        & (labels_int == 0)
    )

    false_positive = np.sum(
        (predictions_array == 1)
        & (labels_int == 0)
    )

    false_negative = np.sum(
        (predictions_array == 0)
        & (labels_int == 1)
    )

    accuracy = (
        (true_positive + true_negative)
        / total_samples
    )

    precision_denominator = (
        true_positive + false_positive
    )

    recall_denominator = (
        true_positive + false_negative
    )

    precision = (
        true_positive / precision_denominator
        if precision_denominator > 0
        else 0.0
    )

    recall = (
        true_positive / recall_denominator
        if recall_denominator > 0
        else 0.0
    )

    f1_denominator = precision + recall

    f1 = (
        2.0 * precision * recall / f1_denominator
        if f1_denominator > 0
        else 0.0
    )

    roc_auc = _binary_roc_auc(
        labels_int,
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


def _binary_roc_auc(
    labels: np.ndarray,
    probabilities: np.ndarray,
) -> float:
    """
    Calculate binary ROC-AUC without requiring the full sklearn
    metric stack inside the training loop.

    Returns 0.5 when only one class is present because ROC-AUC
    is undefined in that situation.
    """
    positive_count = np.sum(labels == 1)
    negative_count = np.sum(labels == 0)

    if positive_count == 0 or negative_count == 0:
        return 0.5

    order = np.argsort(
        probabilities,
        kind="mergesort",
    )

    sorted_labels = labels[order]

    positive_ranks = (
        np.flatnonzero(sorted_labels == 1) + 1
    )

    rank_sum = positive_ranks.sum()

    auc = (
        rank_sum
        - positive_count * (positive_count + 1) / 2
    ) / (positive_count * negative_count)

    return float(auc)
