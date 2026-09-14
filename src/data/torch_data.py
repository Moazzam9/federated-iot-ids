from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import (
    EXPECTED_FEATURE_COUNT,
    FittedStandardScaler,
)


@dataclass(frozen=True)
class BatchData:
    """A single PyTorch batch for binary intrusion detection."""

    features: torch.Tensor
    labels: torch.Tensor


def dataframe_to_tensors(
    features: pd.DataFrame,
    labels: pd.Series,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Convert a feature DataFrame and binary-label Series into PyTorch tensors.
    """
    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"but received {features.shape[1]}."
        )

    if len(features) != len(labels):
        raise ValueError(
            "Features and labels must contain the same number of rows."
        )

    feature_array = features.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    label_array = labels.to_numpy(
        dtype=np.float32,
        copy=True,
    )

    x = torch.from_numpy(feature_array)
    y = torch.from_numpy(label_array).reshape(-1, 1)

    return x, y


def make_dataloader(
    features: pd.DataFrame,
    labels: pd.Series,
    batch_size: int = 256,
    shuffle: bool = False,
) -> DataLoader:
    """
    Create a PyTorch DataLoader from a single in-memory chunk.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    x, y = dataframe_to_tensors(
        features,
        labels,
    )

    dataset = TensorDataset(x, y)

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
    )


def transform_chunk(
    scaler: FittedStandardScaler,
    features: pd.DataFrame,
    labels: pd.Series,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> Iterator[BatchData]:
    """
    Transform one already-loaded data chunk and yield PyTorch batches.

    The scaler must have been fitted previously on training data.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    if device is not None and device != "cpu":
        raise ValueError(
            "Only CPU execution is currently supported."
        )

    transformed_features = scaler.transform(features)

    batch_loader = make_dataloader(
        transformed_features,
        labels,
        batch_size=batch_size,
        shuffle=False,
    )

    for batch_features, batch_labels in batch_loader:
        if device is not None:
            batch_features = batch_features.to(device)
            batch_labels = batch_labels.to(device)

        yield BatchData(
            features=batch_features,
            labels=batch_labels,
        )


def iter_torch_batches(
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
    split: str,
    devices: Optional[list[str]] = None,
    attack_families: Optional[list[str]] = None,
    batch_size: int = 256,
    device: Optional[str] = None,
) -> Iterator[BatchData]:
    """
    Stream N-BaIoT data through the preprocessing and PyTorch pipeline.

    The N-BaIoT loader reads frozen split-indexed CSV chunks. Each chunk
    is transformed using a scaler that must have been fitted previously
    on training data only. The transformed chunk is then converted into
    PyTorch batches.

    This function never loads the complete selected split into memory.
    """
    if batch_size <= 0:
        raise ValueError(
            "batch_size must be greater than zero."
        )

    for data_chunk in loader.iter_chunks(
        split=split,
        devices=devices,
        attack_families=attack_families,
    ):
        yield from transform_chunk(
            scaler=scaler,
            features=data_chunk.features,
            labels=data_chunk.binary_labels,
            batch_size=batch_size,
            device=device,
        )
