from __future__ import annotations

from typing import Iterator, Protocol

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.data.torch_data import BatchData
from src.fl.iid_data import IIDClientDataLoader


class ClientDataSource(Protocol):
    """
    Interface used by federated local training.

    A client data source must provide:
        1. the number of training samples assigned to a client;
        2. standardized PyTorch batches for that client.

    This allows the same local-training implementation to support
    different client-partitioning strategies.
    """

    def count_training_samples(
        self,
        client_id: str,
    ) -> int:
        """Return the number of training samples assigned to a client."""
        ...

    def iter_torch_batches(
        self,
        client_id: str,
        scaler: FittedStandardScaler,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> Iterator[BatchData]:
        """Yield standardized PyTorch batches for one client."""
        ...


class DeviceClientDataSource:
    """
    Adapter for the original device-based N-BaIoT client definition.

    Each physical device represented in N-BaIoT is treated as one
    simulated logical federated client.
    """

    def __init__(
        self,
        loader: NBaIoTSplitLoader,
    ) -> None:
        self.loader = loader

    def count_training_samples(
        self,
        client_id: str,
    ) -> int:
        """Return the frozen training-row count for one device client."""

        if not client_id:
            raise ValueError(
                "client_id must not be empty."
            )

        total_samples = sum(
            record.train_count
            for record in self.loader.source_records
            if record.device == client_id
        )

        if total_samples <= 0:
            raise ValueError(
                f"Client {client_id!r} has no training samples."
            )

        return total_samples

    def iter_torch_batches(
        self,
        client_id: str,
        scaler: FittedStandardScaler,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> Iterator[BatchData]:
        """Yield standardized batches for one device client."""

        yield from self.loader.iter_torch_batches(
            scaler=scaler,
            split="train",
            devices=[client_id],
            batch_size=batch_size,
            device=device,
        )


class IIDClientDataSource:
    """
    Adapter for the deterministic stratified IID client partition.

    Client IDs use:
        client_1
        client_2
        ...
        client_9
    """

    def __init__(
        self,
        loader: IIDClientDataLoader,
    ) -> None:
        self.loader = loader

    def count_training_samples(
        self,
        client_id: str,
    ) -> int:
        """Return the number of training rows assigned to one IID client."""

        return self.loader.count_client_rows(
            client_id
        )

    def iter_torch_batches(
        self,
        client_id: str,
        scaler: FittedStandardScaler,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> Iterator[BatchData]:
        """Yield standardized batches for one IID client."""

        yield from self.loader.iter_torch_batches(
            client_id=client_id,
            scaler=scaler,
            batch_size=batch_size,
            device=device,
        )