from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import Iterator, Optional

import torch
from torch import nn

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.data.torch_data import BatchData, iter_torch_batches
from src.models.trainer import TrainingMetrics, train_one_epoch
from src.fl.client_data import (
    ClientDataSource,
    DeviceClientDataSource,
)


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
    batches: Iterator[BatchData],
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


def _count_client_training_samples(
    loader: NBaIoTSplitLoader,
    client_id: str,
) -> int:
    """
    Return the number of training examples assigned to one
    device-based N-BaIoT client.

    This compatibility helper preserves the existing behavior
    used by the current tests while delegating counting to the
    device client data source.
    """

    data_source = DeviceClientDataSource(loader)

    return data_source.count_training_samples(
        client_id
    )


def _train_device_client_batches(
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
    client_id: str,
    batch_size: int,
    device: str,
) -> Iterator[BatchData]:
    """
    Yield batches for a device client.

    This function deliberately uses the module-level
    `iter_torch_batches` symbol so the existing real-data smoke
    tests can continue to monkeypatch it.

    IID clients use their own data-source implementation.
    """

    yield from iter_torch_batches(
        loader=loader,
        scaler=scaler,
        split="train",
        devices=[client_id],
        batch_size=batch_size,
        device=device,
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
    client_data_source: Optional[ClientDataSource] = None,
) -> ClientTrainingResult:
    """
    Train one federated client using the supplied client data source.

    If no data source is supplied, the original device-based
    N-BaIoT client definition is used.

    IID experiments can supply an IIDClientDataSource.

    `samples` represents the number of training examples available
    to the client. It is not multiplied by the number of local
    epochs because FedAvg weighting is based on client dataset size.
    """

    if not client_id:
        raise ValueError(
            "client_id must not be empty."
        )

    if epochs <= 0:
        raise ValueError(
            "epochs must be greater than zero."
        )

    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if learning_rate <= 0:
        raise ValueError(
            "learning_rate must be greater than zero."
        )

    if device != "cpu":
        raise ValueError(
            "Only CPU execution is currently supported."
        )

    if client_data_source is None:
        client_data_source = DeviceClientDataSource(
            loader
        )

        client_samples = (
            client_data_source.count_training_samples(
                client_id
            )
        )

        def batch_source() -> Iterator[BatchData]:
            yield from _train_device_client_batches(
                loader=loader,
                scaler=scaler,
                client_id=client_id,
                batch_size=batch_size,
                device=device,
            )

    else:
        client_samples = (
            client_data_source.count_training_samples(
                client_id
            )
        )

        def batch_source() -> Iterator[BatchData]:
            yield from (
                client_data_source.iter_torch_batches(
                    client_id=client_id,
                    scaler=scaler,
                    batch_size=batch_size,
                    device=device,
                )
            )

    criterion = nn.BCELoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate,
    )

    epoch_metrics: list[TrainingMetrics] = []

    for _ in range(epochs):
        metrics = train_one_epoch(
            model=model,
            batches=_batch_iterator(
                batch_source()
            ),
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )

        epoch_metrics.append(metrics)

    total_elapsed = sum(
        metrics.elapsed_seconds
        for metrics in epoch_metrics
    )

    total_epoch_samples = sum(
        metrics.samples
        for metrics in epoch_metrics
    )

    if total_epoch_samples <= 0:
        raise ValueError(
            f"Client {client_id!r} produced no training samples."
        )

    weighted_loss = sum(
        metrics.loss * metrics.samples
        for metrics in epoch_metrics
    ) / total_epoch_samples

    return ClientTrainingResult(
        client_id=client_id,
        state_dict=_clone_state_dict_to_cpu(model),
        samples=client_samples,
        epochs=epochs,
        loss=float(weighted_loss),
        elapsed_seconds=float(total_elapsed),
    )