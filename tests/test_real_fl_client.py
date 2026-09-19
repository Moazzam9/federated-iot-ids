from __future__ import annotations

from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.data.torch_data import iter_torch_batches
from src.fl.client import train_client
from src.models.mlp import SmallMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "federated-iot-temp"

DEVICE = "Danmini_Doorbell"


@pytest.fixture(scope="module")
def nbaiot_loader() -> NBaIoTSplitLoader:
    loader = NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        data_root=DATA_ROOT,
        chunk_size=50_000,
    )

    loader.validate_source_indexes()

    return loader


@pytest.fixture(scope="module")
def training_scaler(
    nbaiot_loader: NBaIoTSplitLoader,
) -> FittedStandardScaler:
    """
    Load the official training-only scaler artifact.

    The scaler must already exist because this test exercises the
    real preprocessing pipeline rather than creating a new artifact.
    """
    scaler_path = (
        PROJECT_ROOT
        / "data"
        / "processed"
        / "preprocessing"
        / "training_standard_scaler.pkl"
    )

    if not scaler_path.exists():
        pytest.skip(
            "Official training scaler artifact does not exist."
        )

    import pickle

    with scaler_path.open("rb") as handle:
        scaler = pickle.load(handle)

    return scaler


def _expected_device_training_samples(
    loader: NBaIoTSplitLoader,
    device: str,
) -> int:
    """
    Return the complete frozen training-row count for one device.

    This uses the validated source-index metadata rather than the
    limited smoke-test batch size.
    """
    total = sum(
        record.train_count
        for record in loader.source_records
        if record.device == device
    )

    if total <= 0:
        raise AssertionError(
            f"No training samples found for device {device!r}."
        )

    return total


def test_real_nbaiot_client_trains_one_small_batch(
    nbaiot_loader: NBaIoTSplitLoader,
    training_scaler: FittedStandardScaler,
    monkeypatch,
) -> None:
    """
    Verify that the federated client trainer can train a SmallMLP
    using a real N-BaIoT batch belonging to one logical device.

    Only one real batch is supplied to the training loop, so this
    remains a smoke test rather than a full experiment.

    The returned `samples` value represents the client's complete
    training dataset size because that value is used for FedAvg
    weighting. It is therefore intentionally larger than the
    single smoke-test batch.
    """

    import src.fl.client as client_module

    requested_devices = []

    real_batches = iter_torch_batches(
        loader=nbaiot_loader,
        scaler=training_scaler,
        split="train",
        devices=[DEVICE],
        batch_size=256,
    )

    first_batch = next(real_batches)

    assert first_batch.features.shape[1] == 115
    assert first_batch.labels.shape[1] == 1
    assert first_batch.features.shape[0] <= 256

    def limited_batches(
        loader,
        scaler,
        split,
        devices=None,
        attack_families=None,
        batch_size=256,
        device=None,
    ):
        requested_devices.append(devices)

        yield first_batch

    monkeypatch.setattr(
        client_module,
        "iter_torch_batches",
        limited_batches,
    )

    model = SmallMLP()

    before = OrderedDict(
        (
            name,
            tensor.detach().clone(),
        )
        for name, tensor in model.state_dict().items()
    )

    result = train_client(
        model=model,
        loader=nbaiot_loader,
        scaler=training_scaler,
        client_id=DEVICE,
        epochs=1,
        batch_size=256,
        learning_rate=0.001,
        device="cpu",
    )

    assert requested_devices == [[DEVICE]]

    assert result.client_id == DEVICE

    expected_samples = _expected_device_training_samples(
        loader=nbaiot_loader,
        device=DEVICE,
    )

    assert result.samples == expected_samples
    assert result.samples == 712_809

    assert result.epochs == 1
    assert result.loss >= 0.0
    assert result.elapsed_seconds >= 0.0

    assert set(result.state_dict.keys()) == set(
        before.keys()
    )

    changed = any(
        not torch.equal(
            before[name],
            result.state_dict[name],
        )
        for name in before
    )

    assert changed


def test_real_nbaiot_client_result_is_cpu(
    nbaiot_loader: NBaIoTSplitLoader,
    training_scaler: FittedStandardScaler,
    monkeypatch,
) -> None:
    """
    Verify that the real N-BaIoT client returns a CPU state dict.

    Only one real batch is supplied to the training loop, so this
    remains a smoke test.
    """

    import src.fl.client as client_module

    real_batches = iter_torch_batches(
        loader=nbaiot_loader,
        scaler=training_scaler,
        split="train",
        devices=[DEVICE],
        batch_size=256,
    )

    first_batch = next(real_batches)

    def limited_batches(
        loader,
        scaler,
        split,
        devices=None,
        attack_families=None,
        batch_size=256,
        device=None,
    ):
        yield first_batch

    monkeypatch.setattr(
        client_module,
        "iter_torch_batches",
        limited_batches,
    )

    model = SmallMLP()

    result = train_client(
        model=model,
        loader=nbaiot_loader,
        scaler=training_scaler,
        client_id=DEVICE,
        epochs=1,
        batch_size=256,
        learning_rate=0.001,
        device="cpu",
    )

    for tensor in result.state_dict.values():
        assert tensor.device.type == "cpu"
        assert tensor.requires_grad is False
