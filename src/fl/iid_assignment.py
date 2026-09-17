from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np

from src.fl.partitioning import ClientPartition


def partitions_to_assignment_array(
    partitions: Iterable[ClientPartition],
    total_rows: int,
) -> np.ndarray:
    """
    Convert client partitions into a compact row-to-client assignment array.

    The array is indexed by the experiment-level global training-row ID.

    Example:

        assignment[0] == 3

    means global training row 0 belongs to client_4.

    Stored values are zero-based client indexes:

        0 -> client_1
        1 -> client_2
        ...
        8 -> client_9
    """

    if total_rows <= 0:
        raise ValueError(
            "total_rows must be greater than zero."
        )

    partitions = list(partitions)

    if not partitions:
        raise ValueError(
            "At least one client partition is required."
        )

    if len(partitions) != len(
        {partition.client_id for partition in partitions}
    ):
        raise ValueError(
            "Client IDs must be unique."
        )

    assignment = np.full(
        total_rows,
        -1,
        dtype=np.int8,
    )

    for client_index, partition in enumerate(partitions):

        row_indices = np.asarray(
            partition.row_indices,
            dtype=np.int64,
        )

        if row_indices.ndim != 1:
            raise ValueError(
                f"Row indices for {partition.client_id!r} "
                "must be one-dimensional."
            )

        if len(np.unique(row_indices)) != len(row_indices):
            raise ValueError(
                f"Partition {partition.client_id!r} "
                "contains duplicate row indices."
            )

        if len(row_indices) == 0:
            continue

        if np.any(row_indices < 0):
            raise ValueError(
                f"Partition {partition.client_id!r} "
                "contains negative row indices."
            )

        if np.any(row_indices >= total_rows):
            raise ValueError(
                f"Partition {partition.client_id!r} "
                "contains row indices outside the expected range."
            )

        if np.any(assignment[row_indices] != -1):
            raise ValueError(
                "At least one row is assigned to multiple clients."
            )

        assignment[row_indices] = client_index

    unassigned = np.flatnonzero(
        assignment == -1
    )

    if len(unassigned) > 0:
        raise ValueError(
            "Some training rows were not assigned to a client.\n"
            f"Unassigned rows: {len(unassigned):,}"
        )

    return assignment


def save_assignment_array(
    assignment: np.ndarray,
    path: Path,
) -> None:
    """
    Save a validated IID assignment array as a NumPy .npy file.
    """

    assignment = np.asarray(
        assignment,
        dtype=np.int8,
    )

    if assignment.ndim != 1:
        raise ValueError(
            "assignment must be one-dimensional."
        )

    if len(assignment) == 0:
        raise ValueError(
            "assignment must not be empty."
        )

    if np.any(assignment < 0):
        raise ValueError(
            "assignment contains unassigned rows."
        )

    path = Path(path)
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    np.save(
        path,
        assignment,
        allow_pickle=False,
    )


def load_assignment_array(
    path: Path,
    expected_rows: int,
    expected_clients: int,
) -> np.ndarray:
    """
    Load and validate a saved IID assignment array.
    """

    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(
            "IID assignment file does not exist:\n"
            f"{path}"
        )

    assignment = np.load(
        path,
        allow_pickle=False,
    )

    if assignment.ndim != 1:
        raise ValueError(
            "IID assignment array must be one-dimensional."
        )

    if len(assignment) != expected_rows:
        raise ValueError(
            "IID assignment row count mismatch.\n"
            f"Expected: {expected_rows:,}\n"
            f"Actual: {len(assignment):,}"
        )

    if np.any(assignment < 0):
        raise ValueError(
            "IID assignment contains unassigned rows."
        )

    if np.any(assignment >= expected_clients):
        raise ValueError(
            "IID assignment contains an invalid client index."
        )

    return assignment