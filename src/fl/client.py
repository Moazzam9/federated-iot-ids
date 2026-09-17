from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator

import torch
from torch import nn

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.data.torch_data import iter_torch_batches
from src.models.trainer import TrainingMetrics, train_one_epoch


@dataclass(frozen=True)
class ClientTrainingResult:
    """Result returned after one client's local training."""

    client_id: str
    state_dict: OrderedDict[str, torch.Tensor]
    samples: int
    epochs: int
    loss: float
    elapsed_seconds: float


def _batch_iterator(
    batches,
) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
    """Convert BatchData objects into trainer-compatible tuples."""
    for batch in batches:
        yield batch.features, batch.labels


def _clone_state_dict_to_cpu(
    model: nn.Module,
) -> OrderedDict[str, torch.Tensor]:
    """
    Copy model state to CPU so the client result is independent
    of the client's model object.
    """
    return OrderedDict(
        (
            name,
            tensor.detach().cpu().clone(),
        )
        for name, tensor in model.state_dict().items()
    )


def train_client(
    model: nn.Module,
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
    client_id: str,
    epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    device: str = "cpu",
) -> ClientTrainingResult:
    """
    Train one federated client on only its assigned N-BaIoT data.

    The client receives the current global model through `model`.
    A new optimizer is created for this local training round.

    For device-based clients, `client_id` must be a valid N-BaIoT
    device name. The loader's devices=[client_id] filter ensures that
    only that device's rows are used.
    """
    if not client_id:
        raise ValueError("client_id must not be empty.")

    if epochs <= 0:
        raise ValueError("epochs must be greater than zero.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")

    if device != "cpu":
        raise ValueError(
            "Only CPU execution is currently supported."
        )

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    epoch_metrics: list[TrainingMetrics] = []

    for _ in range(epochs):
        batches = iter_torch_batches(
            loader=loader,
            scaler=scaler,
            split="train",
            devices=[client_id],
            batch_size=batch_size,
            device=device,
        )

        metrics = train_one_epoch(
            model=model,
            batches=_batch_iterator(batches),
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        epoch_metrics.append(metrics)

    total_samples = sum(
        metrics.samples
        for metrics in epoch_metrics
    )

    total_elapsed = sum(
        metrics.elapsed_seconds
        for metrics in epoch_metrics
    )

    weighted_loss = sum(
        metrics.loss * metrics.samples
        for metrics in epoch_metrics
    ) / total_samples

    return ClientTrainingResult(
        client_id=client_id,
        state_dict=_clone_state_dict_to_cpu(model),
        samples=total_samples,
        epochs=epochs,
        loss=float(weighted_loss),
        elapsed_seconds=float(total_elapsed),
    )