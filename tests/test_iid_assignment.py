from pathlib import Path

import numpy as np
import pytest

from src.fl.iid_assignment import (
    load_assignment_array,
    partitions_to_assignment_array,
    save_assignment_array,
)
from src.fl.partitioning import ClientPartition


def test_partitions_to_assignment_array():
    partitions = [
        ClientPartition(
            client_id="client_1",
            row_indices=np.array([0, 2, 4]),
        ),
        ClientPartition(
            client_id="client_2",
            row_indices=np.array([1, 3, 5]),
        ),
    ]

    assignment = partitions_to_assignment_array(
        partitions=partitions,
        total_rows=6,
    )

    expected = np.array(
        [0, 1, 0, 1, 0, 1],
        dtype=np.int8,
    )

    np.testing.assert_array_equal(
        assignment,
        expected,
    )


def test_assignment_requires_complete_coverage():
    partitions = [
        ClientPartition(
            client_id="client_1",
            row_indices=np.array([0, 1]),
        ),
    ]

    with pytest.raises(ValueError, match="not assigned"):
        partitions_to_assignment_array(
            partitions=partitions,
            total_rows=3,
        )


def test_assignment_rejects_overlap():
    partitions = [
        ClientPartition(
            client_id="client_1",
            row_indices=np.array([0, 1]),
        ),
        ClientPartition(
            client_id="client_2",
            row_indices=np.array([1, 2]),
        ),
    ]

    with pytest.raises(ValueError, match="multiple clients"):
        partitions_to_assignment_array(
            partitions=partitions,
            total_rows=3,
        )


def test_assignment_save_and_load(tmp_path: Path):
    assignment = np.array(
        [0, 1, 0, 1, 0, 1],
        dtype=np.int8,
    )

    path = tmp_path / "assignment.npy"

    save_assignment_array(
        assignment=assignment,
        path=path,
    )

    loaded = load_assignment_array(
        path=path,
        expected_rows=6,
        expected_clients=2,
    )

    np.testing.assert_array_equal(
        loaded,
        assignment,
    )


def test_assignment_rejects_wrong_row_count(tmp_path: Path):
    assignment = np.array(
        [0, 1, 0],
        dtype=np.int8,
    )

    path = tmp_path / "assignment.npy"

    save_assignment_array(
        assignment=assignment,
        path=path,
    )

    with pytest.raises(
        ValueError,
        match="row count mismatch",
    ):
        load_assignment_array(
            path=path,
            expected_rows=4,
            expected_clients=2,
        )


def test_assignment_rejects_invalid_client_index(tmp_path: Path):
    assignment = np.array(
        [0, 1, 2],
        dtype=np.int8,
    )

    path = tmp_path / "assignment.npy"

    save_assignment_array(
        assignment=assignment,
        path=path,
    )

    with pytest.raises(
        ValueError,
        match="invalid client index",
    ):
        load_assignment_array(
            path=path,
            expected_rows=3,
            expected_clients=2,
        )