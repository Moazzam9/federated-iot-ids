from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.fl.client import train_client
from src.models.mlp import SmallMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "federated-iot-temp"

DEVICE = "Danmini_Doorbell"


@pytest.fixture(scope="module")
def nbaiot_loader() -> NBaIoTSplitLoader:
    if not DATA_ROOT.exists():
        pytest.skip(
            f"N-BaIoT data root does not exist: {DATA_ROOT}"
        )

    return NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        data_root=DATA_ROOT,
        chunk_size=5_000,
    )


@pytest.fixture(scope="module")
def training_scaler(
    nbaiot_loader: NBaIoTSplitLoader,
) -> FittedStandardScaler:
    train_chunks = nbaiot_loader.iter_chunks(
        split="train",
        devices=[DEVICE],
    )

    first_chunk = next(train_chunks)

    # This is only a smoke-test scaler.
    #
    # The actual experiment uses the saved scaler fitted on the
    # complete training split. This test only needs a valid scaler
    # so that a real N-BaIoT batch can reach the client trainer.
    from src.data.preprocessing import fit_training_scaler

    return fit_training_scaler(
        first_chunk.features
    )


def test_real_nbaiot_client_trains_one_small_batch(
    nbaiot_loader: NBaIoTSplitLoader,
    training_scaler: FittedStandardScaler,
    monkeypatch,
) -> None:
    """
    Verify that the federated client trainer can train a SmallMLP
    using a real N-BaIoT batch belonging to one logical device.

    Only one real batch is supplied to the client trainer so this
    remains a smoke test rather than a full experiment.
    """

    import src.fl.client as client_module
    from src.data.torch_data import iter_torch_batches

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
    assert result.samples == len(
        first_batch.features
    )
    assert result.epochs == 1
    assert result.loss >= 0.0
    assert result.elapsed_seconds >= 0.0

    assert isinstance(
        result.state_dict,
        OrderedDict,
    )

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
    Verify that returned client parameters are detached CPU tensors.
    """

    import src.fl.client as client_module

    from src.data.torch_data import iter_torch_batches

    real_batches = iter_torch_batches(
        loader=nbaiot_loader,
        scaler=training_scaler,
        split="train",
        devices=[DEVICE],
        batch_size=64,
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
        batch_size=64,
        learning_rate=0.001,
        device="cpu",
    )

    for tensor in result.state_dict.values():
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad