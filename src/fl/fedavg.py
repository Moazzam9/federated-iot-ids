from __future__ import annotations

from collections import OrderedDict
from typing import Mapping, Sequence

import torch
from torch import Tensor


StateDict = OrderedDict[str, Tensor]


def weighted_fedavg(
    client_state_dicts: Sequence[Mapping[str, Tensor]],
    client_sample_counts: Sequence[int],
) -> StateDict:
    """
    Aggregate client model parameters using sample-count weighting.

    For client k:

        weight_k = n_k / sum(n_k)

    and:

        global_parameter =
            sum(weight_k * client_parameter_k)

    where n_k is the number of training samples used by client k.

    All client models must have the same parameter structure and tensor
    shapes.
    """
    if not client_state_dicts:
        raise ValueError("At least one client state_dict is required.")

    if len(client_state_dicts) != len(client_sample_counts):
        raise ValueError(
            "client_state_dicts and client_sample_counts "
            "must have the same length."
        )

    if any(count <= 0 for count in client_sample_counts):
        raise ValueError(
            "All client sample counts must be positive."
        )

    total_samples = sum(client_sample_counts)

    if total_samples <= 0:
        raise ValueError(
            "Total client sample count must be positive."
        )

    reference_keys = list(client_state_dicts[0].keys())

    if not reference_keys:
        raise ValueError(
            "Client state_dicts must not be empty."
        )

    for client_number, state_dict in enumerate(
        client_state_dicts,
        start=1,
    ):
        if list(state_dict.keys()) != reference_keys:
            raise ValueError(
                "All client state_dicts must contain "
                "the same parameter keys. "
                f"Mismatch found at client {client_number}."
            )

    aggregated = OrderedDict()

    for key in reference_keys:
        reference_tensor = client_state_dicts[0][key]

        for client_number, state_dict in enumerate(
            client_state_dicts,
            start=1,
        ):
            tensor = state_dict[key]

            if tensor.shape != reference_tensor.shape:
                raise ValueError(
                    "All client tensors must have matching shapes. "
                    f"Parameter '{key}' has a shape mismatch at "
                    f"client {client_number}."
                )

        if reference_tensor.is_floating_point():
            result = torch.zeros_like(
                reference_tensor,
                dtype=reference_tensor.dtype,
            )

            for state_dict, sample_count in zip(
                client_state_dicts,
                client_sample_counts,
            ):
                weight = sample_count / total_samples

                result += (
                    state_dict[key].to(
                        device=reference_tensor.device,
                        dtype=reference_tensor.dtype,
                    )
                    * weight
                )

            aggregated[key] = result

        else:
            # Non-floating buffers cannot be meaningfully averaged.
            # Preserve the first client's value.
            aggregated[key] = reference_tensor.clone()

    return aggregated