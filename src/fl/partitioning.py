from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import numpy as np


@dataclass(frozen=True)
class ClientPartition:
    """
    Describes which dataset rows belong to one simulated FL client.

    The partition stores row references rather than copying feature data.
    """

    client_id: str
    row_indices: np.ndarray


def make_device_partitions(
    device_names: Iterable[str],
) -> Dict[str, ClientPartition]:
    """
    Create one logical federated client for each IoT device.

    The actual device filtering is performed later by the data loader.
    This function only defines the client identities.
    """
    devices = list(device_names)

    if not devices:
        raise ValueError("At least one device is required.")

    if len(set(devices)) != len(devices):
        raise ValueError("Device names must be unique.")

    return {
        device: ClientPartition(
            client_id=device,
            row_indices=np.empty(0, dtype=np.int64),
        )
        for device in devices
    }


def make_iid_partitions(
    row_indices: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> List[ClientPartition]:
    """
    Deterministically distribute row indices across clients.

    This function balances row counts but does not explicitly stratify
    by class. Use make_stratified_iid_partitions() when class balance
    between clients is required.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")

    row_indices = np.asarray(row_indices, dtype=np.int64)

    if row_indices.ndim != 1:
        raise ValueError("row_indices must be one-dimensional.")

    if len(np.unique(row_indices)) != len(row_indices):
        raise ValueError("row_indices must contain unique values.")

    rng = np.random.default_rng(seed)

    shuffled = row_indices.copy()
    rng.shuffle(shuffled)

    client_arrays = np.array_split(
        shuffled,
        num_clients,
    )

    return [
        ClientPartition(
            client_id=f"client_{client_number + 1}",
            row_indices=client_array.copy(),
        )
        for client_number, client_array in enumerate(client_arrays)
    ]


def make_stratified_iid_partitions(
    row_indices: np.ndarray,
    binary_labels: np.ndarray,
    num_clients: int,
    seed: int = 42,
) -> List[ClientPartition]:
    """
    Create deterministic approximately IID client partitions.

    Rows are stratified by binary class so that every client receives
    approximately the same class proportions as the complete input set.

    Every input row is assigned to exactly one client.
    """
    if num_clients < 1:
        raise ValueError("num_clients must be at least 1.")

    row_indices = np.asarray(
        row_indices,
        dtype=np.int64,
    )

    binary_labels = np.asarray(
        binary_labels,
        dtype=np.int64,
    )

    if row_indices.ndim != 1:
        raise ValueError("row_indices must be one-dimensional.")

    if binary_labels.ndim != 1:
        raise ValueError("binary_labels must be one-dimensional.")

    if len(row_indices) != len(binary_labels):
        raise ValueError(
            "row_indices and binary_labels must have the same length."
        )

    if len(np.unique(row_indices)) != len(row_indices):
        raise ValueError("row_indices must contain unique values.")

    unique_labels = set(
        np.unique(binary_labels).tolist()
    )

    if not unique_labels.issubset({0, 1}):
        raise ValueError(
            "binary_labels must contain only 0 and 1."
        )

    rng = np.random.default_rng(seed)

    client_rows = [
        [] for _ in range(num_clients)
    ]

    for label in [0, 1]:
        class_rows = row_indices[
            binary_labels == label
        ].copy()

        rng.shuffle(class_rows)

        class_splits = np.array_split(
            class_rows,
            num_clients,
        )

        for client_number, split in enumerate(class_splits):
            client_rows[client_number].extend(
                split.tolist()
            )

    partitions = []

    for client_number, rows in enumerate(client_rows):
        rows_array = np.asarray(
            rows,
            dtype=np.int64,
        )

        rng.shuffle(rows_array)

        partitions.append(
            ClientPartition(
                client_id=f"client_{client_number + 1}",
                row_indices=rows_array,
            )
        )

    return partitions


def validate_partitions(
    partitions: Iterable[ClientPartition],
    expected_row_indices: np.ndarray,
) -> None:
    """
    Verify that partitions form a complete, non-overlapping assignment.

    Every expected row must occur exactly once.
    """
    partitions = list(partitions)

    expected = np.asarray(
        expected_row_indices,
        dtype=np.int64,
    )

    if expected.ndim != 1:
        raise ValueError(
            "expected_row_indices must be one-dimensional."
        )

    if len(np.unique(expected)) != len(expected):
        raise ValueError(
            "expected_row_indices must contain unique values."
        )

    assigned_arrays = [
        partition.row_indices
        for partition in partitions
    ]

    if assigned_arrays:
        assigned = np.concatenate(
            assigned_arrays
        )
    else:
        assigned = np.empty(
            0,
            dtype=np.int64,
        )

    if len(np.unique(assigned)) != len(assigned):
        raise ValueError(
            "Partitions contain duplicate row assignments."
        )

    if set(assigned.tolist()) != set(
        expected.tolist()
    ):
        raise ValueError(
            "Partitions do not contain exactly "
            "the expected row indices."
        )