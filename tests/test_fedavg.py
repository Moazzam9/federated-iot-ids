from collections import OrderedDict

import pytest
import torch

from src.fl.fedavg import weighted_fedavg


def make_state(value: float) -> OrderedDict:
    return OrderedDict(
        {
            "weight": torch.tensor(
                [[value, value]],
                dtype=torch.float32,
            ),
            "bias": torch.tensor(
                [value],
                dtype=torch.float32,
            ),
        }
    )


def test_weighted_fedavg_uses_sample_count_weights():
    client_1 = make_state(1.0)
    client_2 = make_state(3.0)

    aggregated = weighted_fedavg(
        client_state_dicts=[client_1, client_2],
        client_sample_counts=[1, 3],
    )

    expected = 2.5

    assert torch.allclose(
        aggregated["weight"],
        torch.tensor([[expected, expected]]),
    )

    assert torch.allclose(
        aggregated["bias"],
        torch.tensor([expected]),
    )


def test_weighted_fedavg_single_client_returns_same_parameters():
    client = make_state(7.0)

    aggregated = weighted_fedavg(
        client_state_dicts=[client],
        client_sample_counts=[100],
    )

    assert torch.equal(
        aggregated["weight"],
        client["weight"],
    )

    assert torch.equal(
        aggregated["bias"],
        client["bias"],
    )


def test_weighted_fedavg_preserves_parameter_keys():
    client_1 = make_state(1.0)
    client_2 = make_state(2.0)

    aggregated = weighted_fedavg(
        client_state_dicts=[client_1, client_2],
        client_sample_counts=[10, 10],
    )

    assert list(aggregated.keys()) == [
        "weight",
        "bias",
    ]


def test_weighted_fedavg_rejects_empty_clients():
    with pytest.raises(ValueError):
        weighted_fedavg(
            client_state_dicts=[],
            client_sample_counts=[],
        )


def test_weighted_fedavg_rejects_mismatched_lengths():
    client = make_state(1.0)

    with pytest.raises(ValueError):
        weighted_fedavg(
            client_state_dicts=[client],
            client_sample_counts=[10, 20],
        )


def test_weighted_fedavg_rejects_non_positive_sample_counts():
    client = make_state(1.0)

    with pytest.raises(ValueError):
        weighted_fedavg(
            client_state_dicts=[client],
            client_sample_counts=[0],
        )


def test_weighted_fedavg_rejects_different_parameter_keys():
    client_1 = make_state(1.0)

    client_2 = OrderedDict(
        {
            "different_weight": torch.tensor(
                [[2.0, 2.0]],
                dtype=torch.float32,
            ),
            "bias": torch.tensor(
                [2.0],
                dtype=torch.float32,
            ),
        }
    )

    with pytest.raises(ValueError):
        weighted_fedavg(
            client_state_dicts=[client_1, client_2],
            client_sample_counts=[10, 10],
        )


def test_weighted_fedavg_rejects_different_parameter_shapes():
    client_1 = make_state(1.0)

    client_2 = OrderedDict(
        {
            "weight": torch.tensor(
                [[2.0, 2.0, 2.0]],
                dtype=torch.float32,
            ),
            "bias": torch.tensor(
                [2.0],
                dtype=torch.float32,
            ),
        }
    )

    with pytest.raises(ValueError):
        weighted_fedavg(
            client_state_dicts=[client_1, client_2],
            client_sample_counts=[10, 10],
        )