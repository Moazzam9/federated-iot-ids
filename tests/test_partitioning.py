import numpy as np
import pytest

from src.fl.partitioning import (
    make_device_partitions,
    make_iid_partitions,
    make_stratified_iid_partitions,
    validate_partitions,
)


def test_make_device_partitions_creates_one_client_per_device():
    devices = [
        "device_a",
        "device_b",
        "device_c",
    ]

    partitions = make_device_partitions(devices)

    assert list(partitions.keys()) == devices

    assert all(
        partition.client_id in devices
        for partition in partitions.values()
    )

    assert all(
        partition.row_indices.dtype == np.int64
        for partition in partitions.values()
    )


def test_make_device_partitions_rejects_empty_devices():
    with pytest.raises(
        ValueError,
        match="At least one device",
    ):
        make_device_partitions([])


def test_make_device_partitions_rejects_duplicate_devices():
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        make_device_partitions(
            [
                "device_a",
                "device_a",
            ]
        )


def test_iid_partitions_assign_every_row_once():
    rows = np.arange(
        100,
        dtype=np.int64,
    )

    partitions = make_iid_partitions(
        rows,
        num_clients=9,
        seed=42,
    )

    validate_partitions(
        partitions,
        rows,
    )

    assigned = np.concatenate(
        [
            partition.row_indices
            for partition in partitions
        ]
    )

    assert len(assigned) == 100
    assert len(np.unique(assigned)) == 100


def test_iid_partitions_are_reproducible():
    rows = np.arange(
        100,
        dtype=np.int64,
    )

    first = make_iid_partitions(
        rows,
        num_clients=9,
        seed=42,
    )

    second = make_iid_partitions(
        rows,
        num_clients=9,
        seed=42,
    )

    for first_partition, second_partition in zip(
        first,
        second,
    ):
        np.testing.assert_array_equal(
            first_partition.row_indices,
            second_partition.row_indices,
        )


def test_iid_partitions_change_with_seed():
    rows = np.arange(
        100,
        dtype=np.int64,
    )

    first = make_iid_partitions(
        rows,
        num_clients=9,
        seed=42,
    )

    second = make_iid_partitions(
        rows,
        num_clients=9,
        seed=123,
    )

    assert any(
        not np.array_equal(
            first_partition.row_indices,
            second_partition.row_indices,
        )
        for first_partition, second_partition in zip(
            first,
            second,
        )
    )


def test_iid_partitions_reject_duplicate_rows():
    rows = np.array(
        [0, 1, 2, 2, 3],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="unique",
    ):
        make_iid_partitions(
            rows,
            num_clients=2,
            seed=42,
        )


def test_validate_partitions_rejects_missing_rows():
    expected = np.arange(
        10,
        dtype=np.int64,
    )

    partitions = make_iid_partitions(
        expected[:9],
        num_clients=3,
        seed=42,
    )

    with pytest.raises(
        ValueError,
        match="exactly",
    ):
        validate_partitions(
            partitions,
            expected,
        )


def test_stratified_iid_partitions_assign_every_row_once():
    rows = np.arange(
        1000,
        dtype=np.int64,
    )

    labels = np.array(
        [0] * 100 + [1] * 900,
        dtype=np.int64,
    )

    partitions = make_stratified_iid_partitions(
        rows,
        labels,
        num_clients=9,
        seed=42,
    )

    validate_partitions(
        partitions,
        rows,
    )

    assigned = np.concatenate(
        [
            partition.row_indices
            for partition in partitions
        ]
    )

    assert len(assigned) == 1000
    assert len(np.unique(assigned)) == 1000


def test_stratified_iid_partitions_have_nearly_equal_class_counts():
    rows = np.arange(
        1000,
        dtype=np.int64,
    )

    labels = np.array(
        [0] * 100 + [1] * 900,
        dtype=np.int64,
    )

    partitions = make_stratified_iid_partitions(
        rows,
        labels,
        num_clients=9,
        seed=42,
    )

    client_distributions = []

    for partition in partitions:
        client_labels = labels[
            partition.row_indices
        ]

        benign = int(
            np.sum(client_labels == 0)
        )

        attack = int(
            np.sum(client_labels == 1)
        )

        client_distributions.append(
            (benign, attack)
        )

    benign_counts = [
        item[0]
        for item in client_distributions
    ]

    attack_counts = [
        item[1]
        for item in client_distributions
    ]

    assert (
        max(benign_counts) - min(benign_counts)
        <= 1
    )

    assert (
        max(attack_counts) - min(attack_counts)
        <= 1
    )


def test_stratified_iid_partitions_preserve_global_class_ratio():
    rows = np.arange(
        1000,
        dtype=np.int64,
    )

    labels = np.array(
        [0] * 100 + [1] * 900,
        dtype=np.int64,
    )

    partitions = make_stratified_iid_partitions(
        rows,
        labels,
        num_clients=9,
        seed=42,
    )

    global_attack_ratio = np.mean(labels)

    for partition in partitions:
        client_labels = labels[
            partition.row_indices
        ]

        client_attack_ratio = np.mean(
            client_labels
        )

        assert (
            abs(
                client_attack_ratio
                - global_attack_ratio
            )
            < 0.01
        )


def test_stratified_iid_partitions_are_reproducible():
    rows = np.arange(
        1000,
        dtype=np.int64,
    )

    labels = np.array(
        [0] * 100 + [1] * 900,
        dtype=np.int64,
    )

    first = make_stratified_iid_partitions(
        rows,
        labels,
        num_clients=9,
        seed=42,
    )

    second = make_stratified_iid_partitions(
        rows,
        labels,
        num_clients=9,
        seed=42,
    )

    for first_partition, second_partition in zip(
        first,
        second,
    ):
        np.testing.assert_array_equal(
            first_partition.row_indices,
            second_partition.row_indices,
        )


def test_stratified_iid_partitions_reject_mismatched_lengths():
    rows = np.arange(
        10,
        dtype=np.int64,
    )

    labels = np.zeros(
        9,
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="same length",
    ):
        make_stratified_iid_partitions(
            rows,
            labels,
            num_clients=3,
            seed=42,
        )


def test_stratified_iid_partitions_reject_invalid_labels():
    rows = np.arange(
        10,
        dtype=np.int64,
    )

    labels = np.array(
        [
            0,
            1,
            0,
            1,
            2,
            0,
            1,
            0,
            1,
            0,
        ],
        dtype=np.int64,
    )

    with pytest.raises(
        ValueError,
        match="only 0 and 1",
    ):
        make_stratified_iid_partitions(
            rows,
            labels,
            num_clients=3,
            seed=42,
        )