from collections import OrderedDict
from pathlib import Path

import pytest
import torch

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.data.torch_data import iter_torch_batches
from src.fl.coordinator import run_federated_round
from src.models.mlp import SmallMLP


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "federated-iot-temp"

CLIENTS = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
]


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
def smoke_scaler(
    nbaiot_loader: NBaIoTSplitLoader,
) -> FittedStandardScaler:
    """
    Fit a temporary scaler on a small real training chunk.

    This is ONLY for the smoke test.

    It is NOT the official experiment scaler. The official experiments
    must use the saved training-only global scaler fitted on all
    4,943,824 training rows.
    """
    first_chunk = next(
        nbaiot_loader.iter_chunks(
            split="train",
            devices=[CLIENTS[0]],
        )
    )

    from src.data.preprocessing import fit_training_scaler

    return fit_training_scaler(
        first_chunk.features
    )


def test_real_three_client_federated_round(
    nbaiot_loader,
    smoke_scaler,
    monkeypatch,
):
    """
    Verify one real three-client FedAvg round using small real batches.

    The production client/coordinator code is not changed for this test.
    Only the batch iterator used by train_client is temporarily limited
    to one real batch per client.
    """
    import src.fl.client as client_module

    requested_clients = []
    captured_client_results = []

    original_iter_torch_batches = iter_torch_batches

    def limited_batches(
        loader,
        scaler,
        split,
        devices=None,
        attack_families=None,
        batch_size=256,
        device=None,
    ):
        assert split == "train"
        assert devices is not None
        assert len(devices) == 1

        client_id = devices[0]
        requested_clients.append(client_id)

        real_batches = original_iter_torch_batches(
            loader=loader,
            scaler=scaler,
            split=split,
            devices=devices,
            attack_families=attack_families,
            batch_size=batch_size,
            device=device,
        )

        first_batch = next(real_batches)

        assert first_batch.features.shape[1] == 115
        assert first_batch.labels.shape[1] == 1
        assert first_batch.features.shape[0] <= batch_size

        yield first_batch

    monkeypatch.setattr(
        client_module,
        "iter_torch_batches",
        limited_batches,
    )

    global_model = SmallMLP()

    before = OrderedDict(
        (
            name,
            tensor.detach().clone(),
        )
        for name, tensor in global_model.state_dict().items()
    )

    result = run_federated_round(
        global_model=global_model,
        loader=nbaiot_loader,
        scaler=smoke_scaler,
        client_ids=CLIENTS,
        local_epochs=1,
        batch_size=64,
        learning_rate=0.001,
        device="cpu",
        round_number=1,
    )

    assert requested_clients == CLIENTS

    assert result.round_number == 1

    assert len(result.client_results) == len(CLIENTS)

    assert result.total_client_samples > 0

    assert result.aggregation_elapsed_seconds >= 0.0

    returned_client_ids = [
        client_result.client_id
        for client_result in result.client_results
    ]

    assert returned_client_ids == CLIENTS

    for client_result in result.client_results:
        assert client_result.samples > 0
        assert client_result.epochs == 1
        assert client_result.loss >= 0.0
        assert client_result.elapsed_seconds >= 0.0

        for tensor in client_result.state_dict.values():
            assert tensor.device.type == "cpu"
            assert not tensor.requires_grad

    assert isinstance(
        result.global_state_dict,
        OrderedDict,
    )

    assert set(result.global_state_dict.keys()) == set(
        before.keys()
    )

    changed = any(
        not torch.equal(
            before[name],
            result.global_state_dict[name],
        )
        for name in before
    )

    assert changed


def test_real_three_client_round_preserves_global_model_state_format(
    nbaiot_loader,
    smoke_scaler,
    monkeypatch,
):
    """
    Verify that a real round returns the complete SmallMLP state
    dictionary with valid tensor shapes and CPU tensors.
    """
    import src.fl.client as client_module

    original_iter_torch_batches = iter_torch_batches

    def limited_batches(
        loader,
        scaler,
        split,
        devices=None,
        attack_families=None,
        batch_size=256,
        device=None,
    ):
        real_batches = original_iter_torch_batches(
            loader=loader,
            scaler=scaler,
            split=split,
            devices=devices,
            attack_families=attack_families,
            batch_size=batch_size,
            device=device,
        )

        yield next(real_batches)

    monkeypatch.setattr(
        client_module,
        "iter_torch_batches",
        limited_batches,
    )

    global_model = SmallMLP()

    result = run_federated_round(
        global_model=global_model,
        loader=nbaiot_loader,
        scaler=smoke_scaler,
        client_ids=CLIENTS,
        local_epochs=1,
        batch_size=32,
        learning_rate=0.001,
        device="cpu",
        round_number=1,
    )

    expected_state = global_model.state_dict()

    assert set(result.global_state_dict.keys()) == set(
        expected_state.keys()
    )

    for name, tensor in result.global_state_dict.items():
        assert tensor.device.type == "cpu"
        assert not tensor.requires_grad
        assert tensor.shape == expected_state[name].shape
        assert torch.isfinite(tensor).all()