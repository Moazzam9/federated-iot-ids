from collections import OrderedDict

import pytest
import torch

from src.fl.client import ClientTrainingResult
from src.fl.coordinator import run_federated_round
from src.models.mlp import SmallMLP


def make_result(
    client_id: str,
    value: float,
    samples: int,
) -> ClientTrainingResult:
    """Create a deterministic fake client result for coordinator tests."""
    model = SmallMLP()

    with torch.no_grad():
        for parameter in model.parameters():
            parameter.fill_(value)

    state_dict = OrderedDict(
        (
            name,
            tensor.detach().cpu().clone(),
        )
        for name, tensor in model.state_dict().items()
    )

    return ClientTrainingResult(
        client_id=client_id,
        state_dict=state_dict,
        samples=samples,
        epochs=1,
        loss=value,
        elapsed_seconds=0.01,
    )


def test_federated_round_weighted_aggregation():
    global_model = SmallMLP()

    calls = []

    def fake_train_client(**kwargs):
        client_id = kwargs["client_id"]
        calls.append(client_id)

        if client_id == "client_1":
            return make_result(
                client_id="client_1",
                value=1.0,
                samples=1,
            )

        if client_id == "client_2":
            return make_result(
                client_id="client_2",
                value=3.0,
                samples=3,
            )

        raise AssertionError(
            f"Unexpected client: {client_id}"
        )

    result = run_federated_round(
        global_model=global_model,
        loader=None,
        scaler=None,
        client_ids=["client_1", "client_2"],
        local_epochs=1,
        batch_size=8,
        learning_rate=0.001,
        client_train_fn=fake_train_client,
        round_number=1,
    )

    assert calls == [
        "client_1",
        "client_2",
    ]

    assert result.round_number == 1
    assert result.total_client_samples == 4

    # FedAvg:
    #
    # (1 * 1 + 3 * 3) / (1 + 3)
    # = 10 / 4
    # = 2.5
    for tensor in result.global_state_dict.values():
        assert torch.allclose(
            tensor,
            torch.full_like(tensor, 2.5),
        )

    assert result.aggregation_elapsed_seconds >= 0.0


def test_federated_round_uses_fresh_local_model_each_client():
    global_model = SmallMLP()

    received_models = []

    def fake_train_client(**kwargs):
        model = kwargs["model"]
        received_models.append(model)

        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)

        return ClientTrainingResult(
            client_id=kwargs["client_id"],
            state_dict=OrderedDict(
                (
                    name,
                    tensor.detach().cpu().clone(),
                )
                for name, tensor in model.state_dict().items()
            ),
            samples=10,
            epochs=1,
            loss=0.1,
            elapsed_seconds=0.01,
        )

    run_federated_round(
        global_model=global_model,
        loader=None,
        scaler=None,
        client_ids=[
            "client_1",
            "client_2",
        ],
        client_train_fn=fake_train_client,
    )

    assert len(received_models) == 2

    # Each client must receive its own independent model.
    assert received_models[0] is not received_models[1]


def test_federated_round_clients_start_from_same_global_state():
    global_model = SmallMLP()

    initial_state = OrderedDict(
        (
            name,
            tensor.detach().clone(),
        )
        for name, tensor in global_model.state_dict().items()
    )

    received_initial_states = []

    def fake_train_client(**kwargs):
        model = kwargs["model"]

        received_initial_states.append(
            OrderedDict(
                (
                    name,
                    tensor.detach().clone(),
                )
                for name, tensor in model.state_dict().items()
            )
        )

        return make_result(
            client_id=kwargs["client_id"],
            value=1.0,
            samples=10,
        )

    run_federated_round(
        global_model=global_model,
        loader=None,
        scaler=None,
        client_ids=[
            "client_1",
            "client_2",
        ],
        client_train_fn=fake_train_client,
    )

    assert len(received_initial_states) == 2

    for name in initial_state:
        assert torch.equal(
            received_initial_states[0][name],
            initial_state[name],
        )

        assert torch.equal(
            received_initial_states[1][name],
            initial_state[name],
        )


def test_federated_round_rejects_duplicate_clients():
    with pytest.raises(
        ValueError,
        match="unique",
    ):
        run_federated_round(
            global_model=SmallMLP(),
            loader=None,
            scaler=None,
            client_ids=[
                "client_1",
                "client_1",
            ],
        )


def test_federated_round_rejects_empty_clients():
    with pytest.raises(
        ValueError,
        match="At least one",
    ):
        run_federated_round(
            global_model=SmallMLP(),
            loader=None,
            scaler=None,
            client_ids=[],
        )


def test_federated_round_rejects_zero_client_samples():
    def fake_train_client(**kwargs):
        return make_result(
            client_id=kwargs["client_id"],
            value=1.0,
            samples=0,
        )

    with pytest.raises(
        ValueError,
        match="no training samples",
    ):
        run_federated_round(
            global_model=SmallMLP(),
            loader=None,
            scaler=None,
            client_ids=["client_1"],
            client_train_fn=fake_train_client,
        )


def test_federated_round_updates_global_model():
    global_model = SmallMLP()

    before = OrderedDict(
        (
            name,
            tensor.detach().clone(),
        )
        for name, tensor in global_model.state_dict().items()
    )

    def fake_train_client(**kwargs):
        return make_result(
            client_id=kwargs["client_id"],
            value=5.0,
            samples=10,
        )

    result = run_federated_round(
        global_model=global_model,
        loader=None,
        scaler=None,
        client_ids=["client_1"],
        client_train_fn=fake_train_client,
    )

    changed = any(
        not torch.equal(
            before[name],
            result.global_state_dict[name],
        )
        for name in before
    )

    assert changed

    # The actual global model should also have been updated.
    for name, tensor in global_model.state_dict().items():
        assert torch.equal(
            tensor.cpu(),
            result.global_state_dict[name],
        )