from __future__ import annotations

import json
from pathlib import Path

import torch

from src.models.mlp import SmallMLP, count_trainable_parameters


# ----------------------------------------------------------------------
# Project paths
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = PROJECT_ROOT / "results" / "raw" / "communication"
OUTPUT_PATH = OUTPUT_DIR / "communication_measurement.json"


# ----------------------------------------------------------------------
# Experiment configuration
# ----------------------------------------------------------------------

CLIENT_COUNT = 9
MEASURED_ROUNDS = 3

# The project configuration allows up to 10 federated rounds.
CONFIGURED_MAX_ROUNDS = 10


# ----------------------------------------------------------------------
# Measurement helpers
# ----------------------------------------------------------------------


def calculate_state_dict_payload_bytes(model: torch.nn.Module) -> int:
    """
    Calculate the raw tensor payload size of the model state_dict.

    This represents the bytes required to transmit the model tensors
    without protocol headers, serialization metadata, compression,
    encryption overhead, or transport-layer overhead.
    """
    total_bytes = 0

    for tensor in model.state_dict().values():
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(
                "Expected every state_dict value to be a torch.Tensor, "
                f"but found {type(tensor).__name__}."
            )

        total_bytes += tensor.numel() * tensor.element_size()

    return total_bytes


def calculate_parameter_count(model: torch.nn.Module) -> int:
    """Return the number of trainable parameters."""
    return count_trainable_parameters(model)


def bytes_to_mib(value: int | float) -> float:
    """Convert bytes to mebibytes."""
    return value / (1024 * 1024)


def bytes_to_mb(value: int | float) -> float:
    """Convert bytes to decimal megabytes."""
    return value / 1_000_000


# ----------------------------------------------------------------------
# Main measurement
# ----------------------------------------------------------------------


def main() -> None:
    model = SmallMLP()

    parameter_count = calculate_parameter_count(model)
    model_payload_bytes = calculate_state_dict_payload_bytes(model)

    # One client receives one global model and sends one updated model
    # in each fully-participating FedAvg round.
    download_bytes_per_client_per_round = model_payload_bytes
    upload_bytes_per_client_per_round = model_payload_bytes

    total_download_bytes_per_round = (
        download_bytes_per_client_per_round * CLIENT_COUNT
    )

    total_upload_bytes_per_round = (
        upload_bytes_per_client_per_round * CLIENT_COUNT
    )

    total_bytes_per_round = (
        total_download_bytes_per_round + total_upload_bytes_per_round
    )

    # Communication for the actual completed 3-round experiments.
    measured_rounds_total_download_bytes = (
        total_download_bytes_per_round * MEASURED_ROUNDS
    )

    measured_rounds_total_upload_bytes = (
        total_upload_bytes_per_round * MEASURED_ROUNDS
    )

    measured_rounds_total_bytes = (
        total_bytes_per_round * MEASURED_ROUNDS
    )

    # Communication if the configured maximum of 10 rounds were completed.
    configured_max_total_download_bytes = (
        total_download_bytes_per_round * CONFIGURED_MAX_ROUNDS
    )

    configured_max_total_upload_bytes = (
        total_upload_bytes_per_round * CONFIGURED_MAX_ROUNDS
    )

    configured_max_total_bytes = (
        total_bytes_per_round * CONFIGURED_MAX_ROUNDS
    )

    # Record the tensor dtypes actually present in the model state.
    state_dict_dtypes = sorted(
        {
            str(tensor.dtype)
            for tensor in model.state_dict().values()
            if isinstance(tensor, torch.Tensor)
        }
    )

    result = {
        "experiment": "communication_measurement",
        "status": "completed",
        "model": "small_mlp",
        "client_count": CLIENT_COUNT,
        "measured_rounds": MEASURED_ROUNDS,
        "configured_max_federated_rounds": CONFIGURED_MAX_ROUNDS,
        "model_parameter_count": parameter_count,
        "state_dict_dtypes": state_dict_dtypes,
        "model_state_dict_payload_bytes": model_payload_bytes,
        "model_state_dict_payload_kib": model_payload_bytes / 1024,
        "model_state_dict_payload_mib": bytes_to_mib(model_payload_bytes),
        "model_state_dict_payload_mb": bytes_to_mb(model_payload_bytes),
        "per_client_per_round": {
            "download_bytes": download_bytes_per_client_per_round,
            "upload_bytes": upload_bytes_per_client_per_round,
            "total_bytes": (
                download_bytes_per_client_per_round
                + upload_bytes_per_client_per_round
            ),
        },
        "per_round_all_clients": {
            "download_bytes": total_download_bytes_per_round,
            "upload_bytes": total_upload_bytes_per_round,
            "total_bytes": total_bytes_per_round,
            "download_mib": bytes_to_mib(total_download_bytes_per_round),
            "upload_mib": bytes_to_mib(total_upload_bytes_per_round),
            "total_mib": bytes_to_mib(total_bytes_per_round),
        },
        "measured_3_round_experiment": {
            "download_bytes": measured_rounds_total_download_bytes,
            "upload_bytes": measured_rounds_total_upload_bytes,
            "total_bytes": measured_rounds_total_bytes,
            "download_mib": bytes_to_mib(
                measured_rounds_total_download_bytes
            ),
            "upload_mib": bytes_to_mib(
                measured_rounds_total_upload_bytes
            ),
            "total_mib": bytes_to_mib(measured_rounds_total_bytes),
            "download_mb": bytes_to_mb(
                measured_rounds_total_download_bytes
            ),
            "upload_mb": bytes_to_mb(
                measured_rounds_total_upload_bytes
            ),
            "total_mb": bytes_to_mb(measured_rounds_total_bytes),
        },
        "configured_10_round_experiment": {
            "download_bytes": configured_max_total_download_bytes,
            "upload_bytes": configured_max_total_upload_bytes,
            "total_bytes": configured_max_total_bytes,
            "download_mib": bytes_to_mib(
                configured_max_total_download_bytes
            ),
            "upload_mib": bytes_to_mib(
                configured_max_total_upload_bytes
            ),
            "total_mib": bytes_to_mib(configured_max_total_bytes),
            "download_mb": bytes_to_mb(
                configured_max_total_download_bytes
            ),
            "upload_mb": bytes_to_mb(
                configured_max_total_upload_bytes
            ),
            "total_mb": bytes_to_mb(configured_max_total_bytes),
        },
        "measurement_method": {
            "description": (
                "Raw model tensor payload based on the model state_dict. "
                "For each fully participating FedAvg round, every client "
                "is assumed to download one global model and upload one "
                "updated model."
            ),
            "actual_network_transfer": False,
            "network_traffic_measured": False,
            "protocol_headers_included": False,
            "serialization_overhead_included": False,
            "compression_included": False,
            "encryption_overhead_included": False,
            "transport_overhead_included": False,
            "clients_per_round": CLIENT_COUNT,
            "full_client_participation": True,
        },
        "interpretation_notes": [
            (
                "These values are simulated communication-volume estimates "
                "for the local FedAvg experiments, not measurements of "
                "actual network traffic."
            ),
            (
                "The calculation assumes one full global-model download "
                "and one full updated-model upload per participating client "
                "per round."
            ),
            (
                "The measured 3-round totals correspond to the completed "
                "3-round IID and device Non-IID experiments."
            ),
            (
                "The 10-round values are projections from the same per-round "
                "communication assumption; no 10-round experiment was run."
            ),
            (
                "Aggregation_elapsed_seconds in the experiment results is "
                "local CPU aggregation time and is not network communication "
                "time."
            ),
        ],
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(result, file, indent=2)

    print("=" * 72)
    print("COMMUNICATION MEASUREMENT: PASS")
    print("=" * 72)
    print(f"Model:                         {result['model']}")
    print(f"Trainable parameters:         {parameter_count:,}")
    print(f"State dict dtypes:             {', '.join(state_dict_dtypes)}")
    print(
        f"Model payload:                 "
        f"{model_payload_bytes:,} bytes "
        f"({bytes_to_mib(model_payload_bytes):.6f} MiB)"
    )
    print()
    print("Per client per round:")
    print(
        f"  Download:                    "
        f"{download_bytes_per_client_per_round:,} bytes"
    )
    print(
        f"  Upload:                      "
        f"{upload_bytes_per_client_per_round:,} bytes"
    )
    print(
        f"  Total:                       "
        f"{download_bytes_per_client_per_round + upload_bytes_per_client_per_round:,} bytes"
    )
    print()
    print(f"Per round across {CLIENT_COUNT} clients:")
    print(
        f"  Download:                    "
        f"{total_download_bytes_per_round:,} bytes "
        f"({bytes_to_mib(total_download_bytes_per_round):.6f} MiB)"
    )
    print(
        f"  Upload:                      "
        f"{total_upload_bytes_per_round:,} bytes "
        f"({bytes_to_mib(total_upload_bytes_per_round):.6f} MiB)"
    )
    print(
        f"  Total:                       "
        f"{total_bytes_per_round:,} bytes "
        f"({bytes_to_mib(total_bytes_per_round):.6f} MiB)"
    )
    print()
    print("Completed 3-round experiments:")
    print(
        f"  Download:                    "
        f"{measured_rounds_total_download_bytes:,} bytes "
        f"({bytes_to_mib(measured_rounds_total_download_bytes):.6f} MiB)"
    )
    print(
        f"  Upload:                      "
        f"{measured_rounds_total_upload_bytes:,} bytes "
        f"({bytes_to_mib(measured_rounds_total_upload_bytes):.6f} MiB)"
    )
    print(
        f"  Total:                       "
        f"{measured_rounds_total_bytes:,} bytes "
        f"({bytes_to_mib(measured_rounds_total_bytes):.6f} MiB)"
    )
    print()
    print("Configured 10-round projection:")
    print(
        f"  Total:                       "
        f"{configured_max_total_bytes:,} bytes "
        f"({bytes_to_mib(configured_max_total_bytes):.6f} MiB)"
    )
    print()
    print(f"Result JSON: {OUTPUT_PATH}")
    print("=" * 72)


if __name__ == "__main__":
    main()