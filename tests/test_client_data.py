from pathlib import Path

import pytest

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.fl.client_data import (
    DeviceClientDataSource,
    IIDClientDataSource,
)
from src.fl.iid_data import IIDClientDataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(r"D:\federated-iot-temp\nbaiot_duplicate_8f_7x_ma")
SPLIT_ROOT = PROJECT_ROOT / "data" / "processed" / "splits"
IID_ASSIGNMENT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "fl"
    / "iid"
    / "train_client_assignments.npy"
)


def make_loader() -> NBaIoTSplitLoader:
    return NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        chunk_size=50_000,
    )


def test_device_client_data_source_uses_validated_metadata():
    loader = make_loader()
    source = DeviceClientDataSource(loader)

    samples = source.count_training_samples("Danmini_Doorbell")

    assert samples == 712_809


def test_device_client_data_source_rejects_unknown_client():
    loader = make_loader()
    source = DeviceClientDataSource(loader)

    with pytest.raises(ValueError, match="has no training samples"):
        source.count_training_samples("unknown_device")


def test_iid_client_data_source_counts_assigned_rows():
    loader = make_loader()

    iid_loader = IIDClientDataLoader(
        loader=loader,
        assignment_path=IID_ASSIGNMENT_PATH,
        num_clients=9,
    )

    source = IIDClientDataSource(iid_loader)

    counts = [
        source.count_training_samples(f"client_{client_number}")
        for client_number in range(1, 10)
    ]

    assert sum(counts) == 4_943_824
    assert min(counts) >= 549_313
    assert max(counts) <= 549_315


def test_iid_client_data_source_rejects_unknown_client():
    loader = make_loader()

    iid_loader = IIDClientDataLoader(
        loader=loader,
        assignment_path=IID_ASSIGNMENT_PATH,
        num_clients=9,
    )

    source = IIDClientDataSource(iid_loader)

    with pytest.raises(ValueError):
        source.count_training_samples("client_10")