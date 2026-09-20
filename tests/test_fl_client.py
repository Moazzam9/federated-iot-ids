from pathlib import Path

import joblib
import pytest
import src.fl.client

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.fl.client import (
    _count_client_training_samples,
    train_client,
)
from src.fl.client_data import IIDClientDataSource
from src.fl.iid_data import IIDClientDataLoader
from src.models.mlp import SmallMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_ROOT = Path(
    r"D:\federated-iot-temp\nbaiot_duplicate_8f_7x_ma"
)

SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "training_standard_scaler.pkl"
)

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


def load_training_scaler() -> FittedStandardScaler:
    scaler = joblib.load(SCALER_PATH)

    assert isinstance(
        scaler,
        FittedStandardScaler,
    )

    return scaler


def test_client_sample_count_uses_validated_metadata():
    loader = make_loader()

    count = _count_client_training_samples(
        loader=loader,
        client_id="Danmini_Doorbell",
    )

    assert count == 712_809


def test_client_sample_count_rejects_unknown_client():
    loader = make_loader()

    with pytest.raises(
        ValueError,
        match="has no training samples",
    ):
        _count_client_training_samples(
            loader=loader,
            client_id="unknown_device",
        )


def test_real_nbaiot_client_trains_one_small_batch(
    monkeypatch,
):
    loader = make_loader()
    scaler = load_training_scaler()
    model = SmallMLP()

    expected_samples = 712_809

    original_iter_torch_batches = (
        src.fl.client.iter_torch_batches
    )

    def one_batch_only(
        loader,
        scaler,
        split,
        devices,
        batch_size,
        device,
    ):
        batches = original_iter_torch_batches(
            loader=loader,
            scaler=scaler,
            split=split,
            devices=devices,
            batch_size=8,
            device=device,
        )

        yield next(batches)

    monkeypatch.setattr(
        src.fl.client,
        "iter_torch_batches",
        one_batch_only,
    )

    result = train_client(
        model=model,
        loader=loader,
        scaler=scaler,
        client_id="Danmini_Doorbell",
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        device="cpu",
    )

    assert result.client_id == "Danmini_Doorbell"
    assert result.samples == expected_samples
    assert result.samples > 0
    assert result.epochs == 1


def test_real_nbaiot_client_result_is_cpu(
    monkeypatch,
):
    loader = make_loader()
    scaler = load_training_scaler()
    model = SmallMLP()

    original_iter_torch_batches = (
        src.fl.client.iter_torch_batches
    )

    def one_batch_only(
        loader,
        scaler,
        split,
        devices,
        batch_size,
        device,
    ):
        batches = original_iter_torch_batches(
            loader=loader,
            scaler=scaler,
            split=split,
            devices=devices,
            batch_size=8,
            device=device,
        )

        yield next(batches)

    monkeypatch.setattr(
        src.fl.client,
        "iter_torch_batches",
        one_batch_only,
    )

    result = train_client(
        model=model,
        loader=loader,
        scaler=scaler,
        client_id="Danmini_Doorbell",
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        device="cpu",
    )

    assert result.client_id == "Danmini_Doorbell"

    for tensor in result.state_dict.values():
        assert tensor.device.type == "cpu"


def test_real_iid_client_trains_one_small_batch(
    monkeypatch,
):
    loader = make_loader()

    iid_loader = IIDClientDataLoader(
        loader=loader,
        assignment_path=IID_ASSIGNMENT_PATH,
        num_clients=9,
    )

    client_data_source = IIDClientDataSource(
        iid_loader
    )

    scaler = load_training_scaler()
    model = SmallMLP()

    expected_samples = (
        client_data_source.count_training_samples(
            "client_1"
        )
    )

    original_iter_torch_batches = (
        IIDClientDataLoader.iter_torch_batches
    )

    def one_batch_only(
        self,
        client_id,
        scaler,
        batch_size=256,
        device="cpu",
    ):
        batches = original_iter_torch_batches(
            self,
            client_id=client_id,
            scaler=scaler,
            batch_size=8,
            device=device,
        )

        yield next(batches)

    monkeypatch.setattr(
        IIDClientDataLoader,
        "iter_torch_batches",
        one_batch_only,
    )

    result = train_client(
        model=model,
        loader=loader,
        scaler=scaler,
        client_id="client_1",
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        device="cpu",
        client_data_source=client_data_source,
    )

    assert result.client_id == "client_1"
    assert result.samples == expected_samples
    assert result.samples > 0
    assert result.epochs == 1
