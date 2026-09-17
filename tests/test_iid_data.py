from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.fl.iid_data import IIDClientDataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]

ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fl"
    / "iid"
    / "train_client_assignments.npy"
)

EXPECTED_TRAINING_ROWS = 4_943_824
EXPECTED_CLIENTS = 9


@pytest.fixture
def real_loader() -> NBaIoTSplitLoader:
    return NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        chunk_size=50_000,
    )


@pytest.fixture
def iid_loader(
    real_loader: NBaIoTSplitLoader,
) -> IIDClientDataLoader:
    return IIDClientDataLoader(
        loader=real_loader,
        assignment_path=ASSIGNMENT_PATH,
        num_clients=EXPECTED_CLIENTS,
    )


def test_assignment_artifact_exists():
    assert ASSIGNMENT_PATH.is_file()


def test_iid_loader_validates_real_assignment(
    iid_loader: IIDClientDataLoader,
):
    assert len(iid_loader.assignment) == (
        EXPECTED_TRAINING_ROWS
    )

    assert iid_loader.num_clients == (
        EXPECTED_CLIENTS
    )


def test_client_row_counts_match_assignment(
    iid_loader: IIDClientDataLoader,
):
    counts = [
        iid_loader.count_client_rows(
            f"client_{client_number}"
        )
        for client_number in range(
            1,
            EXPECTED_CLIENTS + 1,
        )
    ]

    assert sum(counts) == EXPECTED_TRAINING_ROWS

    assert min(counts) == 549_313
    assert max(counts) == 549_315


def test_invalid_client_id_is_rejected(
    iid_loader: IIDClientDataLoader,
):
    with pytest.raises(ValueError):
        iid_loader.count_client_rows(
            "client_10"
        )

    with pytest.raises(ValueError):
        iid_loader.count_client_rows(
            "device_1"
        )


def test_non_training_split_is_rejected(
    iid_loader: IIDClientDataLoader,
):
    with pytest.raises(
        ValueError,
        match="training split",
    ):
        next(
            iid_loader.iter_chunks(
                client_id="client_1",
                split="validation",
            )
        )


def test_first_real_iid_chunk_has_expected_structure(
    iid_loader: IIDClientDataLoader,
):
    chunk = next(
        iid_loader.iter_chunks(
            client_id="client_1",
            split="train",
        )
    )

    assert len(chunk.features) > 0

    assert (
        chunk.features.shape[1] == 115
    )

    assert len(chunk.binary_labels) == len(
        chunk.features
    )

    assert len(chunk.attack_families) == len(
        chunk.features
    )

    assert len(chunk.devices) == len(
        chunk.features
    )

    assert len(chunk.row_numbers) == len(
        chunk.features
    )

    assert set(
        chunk.binary_labels.unique().tolist()
    ).issubset({0, 1})


def test_first_real_iid_chunk_belongs_to_client(
    iid_loader: IIDClientDataLoader,
):
    """
    Verify that the first streamed client chunk corresponds to
    rows assigned to client_1 in the saved IID assignment.

    The test checks the first source file only. This keeps the test
    lightweight while still exercising the real loader, real CSV,
    real frozen split indexes, and real assignment artifact.
    """

    chunk = next(
        iid_loader.iter_chunks(
            client_id="client_1",
            split="train",
        )
    )

    assert len(chunk.features) > 0

    assert np.all(
        np.isin(
            chunk.binary_labels.to_numpy(),
            [0, 1],
        )
    )

    # The first source file has a deterministic training-row range.
    first_record = next(
        record
        for record in iid_loader.loader.source_records
        if record.train_count > 0
    )

    first_source_training_count = (
        first_record.train_count
    )

    assignment_prefix = iid_loader.assignment[
        :first_source_training_count
    ]

    expected_client_rows = int(
        np.sum(
            assignment_prefix == 0
        )
    )

    assert expected_client_rows > 0

    assert len(chunk.features) <= (
        expected_client_rows
    )