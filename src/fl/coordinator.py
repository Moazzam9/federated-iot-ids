from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from time import perf_counter
from typing import Callable, Sequence

import torch
from torch import nn

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.fl.client import ClientTrainingResult, train_client
from src.fl.fedavg import weighted_fedavg


@dataclass(frozen=True)
class FederatedRoundResult:
    """Results produced by one federated training round."""

    round_number: int
    client_results: tuple[ClientTrainingResult, ...]
    global_state_dict: OrderedDict[str, torch.Tensor]
    total_client_samples: int
    aggregation_elapsed_seconds: float


def _clone_state_dict_to_cpu(
    state_dict: dict[str, torch.Tensor],
) -> OrderedDict[str, torch.Tensor]:
    """Return a detached CPU copy of a model state dictionary."""
    return OrderedDict(
        (
            name,
            tensor.detach().cpu().clone(),
        )
        for name, tensor in state_dict.items()
    )


def _load_state_dict_copy(
    model: nn.Module,
    state_dict: dict[str, torch.Tensor],
) -> None:
    """Load a CPU-cloned state dictionary into a model."""
    model.load_state_dict(
        _clone_state_dict_to_cpu(state_dict),
        strict=True,
    )


def _create_model_like(global_model: nn.Module) -> nn.Module:
    """
    Create a fresh model with the same architecture as the global model.

    The current project uses SmallMLP, whose constructor requires no
    arguments. Keeping this in a helper makes the coordinator easier to
    extend later if model construction becomes configurable.
    """
    return type(global_model)()


def run_federated_round(
    global_model: nn.Module,
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
    client_ids: Sequence[str],
    local_epochs: int = 1,
    batch_size: int = 256,
    learning_rate: float = 0.001,
    device: str = "cpu",
    client_train_fn: Callable[..., ClientTrainingResult] = train_client,
    round_number: int = 1,
) -> FederatedRoundResult:
    """
    Run one complete FedAvg training round.

    Workflow:
        1. Save the current global model state.
        2. Create a fresh copy of the global model for each client.
        3. Train each client locally.
        4. Collect each client's trained state and sample count.
        5. Aggregate client models using weighted FedAvg.
        6. Update the global model with the aggregated state.
        7. Return round-level results.

    Client weights are based on the number of training samples returned
    by each client for the round.
    """
    if not client_ids:
        raise ValueError("At least one client_id is required.")

    if len(set(client_ids)) != len(client_ids):
        raise ValueError("client_ids must be unique.")

    if local_epochs <= 0:
        raise ValueError("local_epochs must be greater than zero.")

    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero.")

    if learning_rate <= 0:
        raise ValueError("learning_rate must be greater than zero.")

    if device != "cpu":
        raise ValueError("Only CPU execution is currently supported.")

    if round_number <= 0:
        raise ValueError("round_number must be greater than zero.")

    global_state = _clone_state_dict_to_cpu(
        global_model.state_dict()
    )

    client_results: list[ClientTrainingResult] = []

    for client_id in client_ids:
        local_model = _create_model_like(global_model)

        _load_state_dict_copy(
            local_model,
            global_state,
        )

        result = client_train_fn(
            model=local_model,
            loader=loader,
            scaler=scaler,
            client_id=client_id,
            epochs=local_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            device=device,
        )

        if result.client_id != client_id:
            raise ValueError(
                f"Client training returned client_id={result.client_id!r}, "
                f"expected {client_id!r}."
            )

        if result.samples <= 0:
            raise ValueError(
                f"Client {client_id!r} returned no training samples."
            )

        client_results.append(result)

    client_state_dicts = [
        result.state_dict
        for result in client_results
    ]

    client_sample_counts = [
        result.samples
        for result in client_results
    ]

    aggregation_start = perf_counter()

    aggregated_state = weighted_fedavg(
        client_state_dicts=client_state_dicts,
        client_sample_counts=client_sample_counts,
    )

    aggregation_elapsed_seconds = (
        perf_counter() - aggregation_start
    )

    _load_state_dict_copy(
        global_model,
        aggregated_state,
    )

    total_client_samples = sum(client_sample_counts)

    return FederatedRoundResult(
        round_number=round_number,
        client_results=tuple(client_results),
        global_state_dict=_clone_state_dict_to_cpu(
            global_model.state_dict()
        ),
        total_client_samples=total_client_samples,
        aggregation_elapsed_seconds=float(
            aggregation_elapsed_seconds
        ),
    )