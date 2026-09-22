from __future__ import annotations

from typing import Iterator, Protocol

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.data.torch_data import BatchData, iter_torch_batches
from src.fl.iid_data import IIDClientDataLoader


class ClientDataSource(Protocol):
    def count_training_samples(self, client_id: str) -> int:
        ...

    def iter_torch_batches(
        self,
        client_id: str,
        scaler: FittedStandardScaler,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> Iterator[BatchData]:
        ...


class DeviceClientDataSource:
    def __init__(self, loader: NBaIoTSplitLoader) -> None:
        self.loader = loader

    def count_training_samples(self, client_id: str) -> int:
        if not client_id:
            raise ValueError("client_id must not be empty.")

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
        if not client_id:
            raise ValueError("client_id must not be empty.")

        yield from iter_torch_batches(
            loader=self.loader,
            scaler=scaler,
            split="train",
            devices=[client_id],
            batch_size=batch_size,
            device=device,
        )


class IIDClientDataSource:
    def __init__(self, loader: IIDClientDataLoader) -> None:
        self.loader = loader

    def count_training_samples(self, client_id: str) -> int:
        return self.loader.count_client_rows(client_id)

    def iter_torch_batches(
        self,
        client_id: str,
        scaler: FittedStandardScaler,
        batch_size: int = 256,
        device: str = "cpu",
    ) -> Iterator[BatchData]:
        yield from self.loader.iter_torch_batches(
            client_id=client_id,
            scaler=scaler,
            batch_size=batch_size,
            device=device,
        )