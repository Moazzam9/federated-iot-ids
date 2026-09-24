
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_RAW = PROJECT_ROOT / "results" / "raw"
RESULTS_PROCESSED = PROJECT_ROOT / "results" / "processed"

CENTRALIZED_PATH = RESULTS_RAW / "centralized" / "centralized_training_history.json"
LOCAL_ONLY_PATH = RESULTS_RAW / "local_only" / "local_only_results.json"
DEVICE_NON_IID_PATH = RESULTS_RAW / "device_non_iid_3round_seed42.json"
IID_PATH = RESULTS_RAW / "iid_3round_seed42.json"

COMMUNICATION_PATH = (
    RESULTS_RAW / "communication" / "communication_measurement.json"
)

RESOURCE_DIR = RESULTS_RAW / "resource_measurement"

CENTRALIZED_RESOURCE_PATH = (
    RESOURCE_DIR / "centralized_3epoch.json"
)
LOCAL_ONLY_RESOURCE_PATH = (
    RESOURCE_DIR / "local_only_3epoch.json"
)
DEVICE_NON_IID_RESOURCE_PATH = (
    RESOURCE_DIR / "device_non_iid_3round_seed42.json"
)
IID_RESOURCE_PATH = (
    RESOURCE_DIR / "iid_3round_seed42.json"
)

OUTPUT_PATH = (
    RESULTS_PROCESSED / "experiment_results_summary.json"
)


# ----------------------------------------------------------------------
# JSON helpers
# ----------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    """Load one JSON object and verify that the top level is a dictionary."""
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected a JSON object in {path}")

    return data


def require_key(
    data: dict[str, Any],
    key: str,
    source_name: str,
) -> Any:
    """Return a required key or raise a clear validation error."""
    if key not in data:
        raise KeyError(
            f"Required key {key!r} is missing from {source_name}."
        )

    return data[key]


def require_equal(
    values: dict[str, Any],
    description: str,
) -> None:
    """Verify that all supplied values are equal."""
    unique_values = set()

    for value in values.values():
        try:
            unique_values.add(json.dumps(value, sort_keys=True))
        except TypeError:
            unique_values.add(repr(value))

    if len(unique_values) != 1:
        details = ", ".join(
            f"{name}={value!r}"
            for name, value in values.items()
        )
        raise ValueError(
            f"Cross-experiment consistency check failed for "
            f"{description}: {details}"
        )


def round_number_key(round_data: dict[str, Any]) -> int:
    """Extract and validate a round number."""
    value = require_key(round_data, "round", "round result")

    if not isinstance(value, int):
        raise ValueError(
            f"Round number must be an integer, got {value!r}."
        )

    return value


# ----------------------------------------------------------------------
# Validation
# ----------------------------------------------------------------------


def validate_common_metadata(
    centralized: dict[str, Any],
    local_only: dict[str, Any],
    device_non_iid: dict[str, Any],
    iid: dict[str, Any],
    communication: dict[str, Any],
) -> None:
    """Validate metadata shared across experiments."""

    require_equal(
        {
            "centralized.dataset": centralized.get("dataset"),
            "local_only.dataset": local_only.get("dataset"),
        },
        "dataset between centralized and local-only",
    )

    require_equal(
        {
            "centralized.model": centralized.get("model"),
            "local_only.model": local_only.get("model"),
        },
        "model between centralized and local-only",
    )

    require_equal(
        {
            "centralized.model_parameter_count": centralized.get(
                "parameter_count"
            ),
            "local_only.model_parameter_count": local_only.get(
                "model_parameter_count"
            ),
            "device_non_iid.model_parameter_count": device_non_iid.get(
                "model_parameter_count"
            ),
            "iid.model_parameter_count": iid.get(
                "model_parameter_count"
            ),
            "communication.model_parameter_count": communication.get(
                "model_parameter_count"
            ),
        },
        "model parameter count",
    )

    require_equal(
        {
            "local_only.client_count": local_only.get("client_count"),
            "device_non_iid.client_count": len(
                require_key(
                    device_non_iid,
                    "client_ids",
                    "device_non_iid",
                )
            ),
            "iid.client_count": len(
                require_key(
                    iid,
                    "client_ids",
                    "iid",
                )
            ),
            "communication.client_count": communication.get(
                "client_count"
            ),
        },
        "client count",
    )

    require_equal(
        {
            "local_only.seed": local_only.get("seed"),
            "device_non_iid.seed": device_non_iid.get("seed"),
            "iid.seed": iid.get("seed"),
        },
        "random seed",
    )

    require_equal(
        {
            "local_only.batch_size": local_only.get("batch_size"),
            "device_non_iid.batch_size": device_non_iid.get("batch_size"),
            "iid.batch_size": iid.get("batch_size"),
        },
        "batch size",
    )

    require_equal(
        {
            "local_only.learning_rate": local_only.get("learning_rate"),
            "device_non_iid.learning_rate": device_non_iid.get(
                "learning_rate"
            ),
            "iid.learning_rate": iid.get("learning_rate"),
        },
        "learning rate",
    )

    require_equal(
        {
            "local_only.device": local_only.get("device"),
            "device_non_iid.device": device_non_iid.get("device"),
            "iid.device": iid.get("device"),
        },
        "training device",
    )


def validate_communication(
    communication: dict[str, Any],
) -> None:
    """Validate the measured communication JSON structure."""

    measurement_method = require_key(
        communication,
        "measurement_method",
        "communication measurement",
    )

    if not isinstance(measurement_method, dict):
        raise ValueError(
            "communication.measurement_method must be a JSON object."
        )

    required_measurement_keys = [
        "actual_network_transfer",
        "network_traffic_measured",
        "protocol_headers_included",
        "serialization_overhead_included",
        "compression_included",
        "encryption_overhead_included",
        "transport_overhead_included",
        "clients_per_round",
        "full_client_participation",
    ]

    for key in required_measurement_keys:
        require_key(
            measurement_method,
            key,
            "communication measurement_method",
        )

    require_key(
        communication,
        "model_parameter_count",
        "communication measurement",
    )

    require_key(
        communication,
        "model_state_dict_payload_bytes",
        "communication measurement",
    )

    measured_3_round = require_key(
        communication,
        "measured_3_round_experiment",
        "communication measurement",
    )

    configured_10_round = require_key(
        communication,
        "configured_10_round_experiment",
        "communication measurement",
    )

    if not isinstance(measured_3_round, dict):
        raise ValueError(
            "communication.measured_3_round_experiment "
            "must be a JSON object."
        )

    if not isinstance(configured_10_round, dict):
        raise ValueError(
            "communication.configured_10_round_experiment "
            "must be a JSON object."
        )

    for key in (
        "download_bytes",
        "upload_bytes",
        "total_bytes",
    ):
        require_key(
            measured_3_round,
            key,
            "communication measured_3_round_experiment",
        )
        require_key(
            configured_10_round,
            key,
            "communication configured_10_round_experiment",
        )


def validate_resource_measurement(
    resource: dict[str, Any],
    source_name: str,
) -> None:
    """Validate one process resource measurement JSON object."""

    require_key(resource, "status", source_name)
    require_key(resource, "exit_code", source_name)
    require_key(resource, "wall_time_seconds", source_name)

    measurement = require_key(
        resource,
        "measurement",
        source_name,
    )

    if not isinstance(measurement, dict):
        raise ValueError(
            f"{source_name}.measurement must be a JSON object."
        )

    require_key(
        measurement,
        "peak_rss_bytes",
        f"{source_name}.measurement",
    )
    require_key(
        measurement,
        "peak_rss_mib",
        f"{source_name}.measurement",
    )
    require_key(
        measurement,
        "sampling_interval_seconds",
        f"{source_name}.measurement",
    )
    require_key(
        measurement,
        "samples_collected",
        f"{source_name}.measurement",
    )

    if resource["status"] != "completed":
        raise ValueError(
            f"{source_name} has unexpected status: "
            f"{resource['status']!r}"
        )

    if resource["exit_code"] != 0:
        raise ValueError(
            f"{source_name} has non-zero exit code: "
            f"{resource['exit_code']!r}"
        )


# ----------------------------------------------------------------------
# Summary extraction
# ----------------------------------------------------------------------


def extract_centralized_summary(
    data: dict[str, Any],
    resource: dict[str, Any],
) -> dict[str, Any]:
    """Extract the centralized experiment summary."""

    history = require_key(
        data,
        "history",
        "centralized training history",
    )

    if not isinstance(history, list) or not history:
        raise ValueError(
            "Centralized history must be a non-empty list."
        )

    best_epoch = require_key(
        data,
        "best_epoch",
        "centralized training history",
    )

    best_validation_f1 = require_key(
        data,
        "best_validation_f1",
        "centralized training history",
    )

    return {
        "status": require_key(
            data,
            "experiment",
            "centralized training history",
        ),
        "dataset": require_key(
            data,
            "dataset",
            "centralized training history",
        ),
        "model": require_key(
            data,
            "model",
            "centralized training history",
        ),
        "seed": require_key(
            data,
            "seed",
            "centralized training history",
        ),
        "epochs": require_key(
            data,
            "epochs",
            "centralized training history",
        ),
        "batch_size": require_key(
            data,
            "batch_size",
            "centralized training history",
        ),
        "parameter_count": require_key(
            data,
            "parameter_count",
            "centralized training history",
        ),
        "total_train_rows": history[-1]["train_samples"],
        "total_validation_rows": history[-1]["validation_samples"],
        "best_epoch": best_epoch,
        "best_validation_f1": best_validation_f1,
        "best_validation_roc_auc": history[best_epoch - 1][
            "validation_roc_auc"
        ],
        "best_validation_accuracy": history[best_epoch - 1][
            "validation_accuracy"
        ],
        "best_validation_precision": history[best_epoch - 1][
            "validation_precision"
        ],
        "best_validation_recall": history[best_epoch - 1][
            "validation_recall"
        ],
        "best_validation_loss": history[best_epoch - 1][
            "validation_loss"
        ],
        "total_elapsed_seconds": require_key(
            data,
            "total_elapsed_seconds",
            "centralized training history",
        ),
        "resource_peak_rss_bytes": resource["measurement"][
            "peak_rss_bytes"
        ],
        "resource_peak_rss_mib": resource["measurement"][
            "peak_rss_mib"
        ],
        "resource_wall_time_seconds": resource[
            "wall_time_seconds"
        ],
    }


def extract_local_only_summary(
    data: dict[str, Any],
    resource: dict[str, Any],
) -> dict[str, Any]:
    """Extract the local-only experiment summary."""

    macro_average = require_key(
        data,
        "macro_average",
        "local-only results",
    )

    sample_weighted_average = require_key(
        data,
        "sample_weighted_average",
        "local-only results",
    )

    return {
        "status": require_key(
            data,
            "status",
            "local-only results",
        ),
        "partition_type": require_key(
            data,
            "partition_type",
            "local-only results",
        ),
        "seed": require_key(
            data,
            "seed",
            "local-only results",
        ),
        "client_count": require_key(
            data,
            "client_count",
            "local-only results",
        ),
        "epochs_per_client": require_key(
            data,
            "epochs_per_client",
            "local-only results",
        ),
        "batch_size": require_key(
            data,
            "batch_size",
            "local-only results",
        ),
        "learning_rate": require_key(
            data,
            "learning_rate",
            "local-only results",
        ),
        "device": require_key(
            data,
            "device",
            "local-only results",
        ),
        "parameter_count": require_key(
            data,
            "model_parameter_count",
            "local-only results",
        ),
        "total_train_rows": require_key(
            data,
            "total_training_rows",
            "local-only results",
        ),
        "total_validation_rows": require_key(
            data,
            "total_validation_rows",
            "local-only results",
        ),
        "macro_average": macro_average,
        "sample_weighted_average": sample_weighted_average,
        "total_elapsed_seconds": require_key(
            data,
            "total_experiment_wall_time_seconds",
            "local-only results",
        ),
        "resource_peak_rss_bytes": resource["measurement"][
            "peak_rss_bytes"
        ],
        "resource_peak_rss_mib": resource["measurement"][
            "peak_rss_mib"
        ],
        "resource_wall_time_seconds": resource[
            "wall_time_seconds"
        ],
    }


def extract_fedavg_summary(
    data: dict[str, Any],
    resource: dict[str, Any],
) -> dict[str, Any]:
    """Extract a multi-round FedAvg experiment summary."""

    rounds = require_key(
        data,
        "rounds",
        "FedAvg results",
    )

    if not isinstance(rounds, list) or not rounds:
        raise ValueError(
            "FedAvg rounds must be a non-empty list."
        )

    last_round = rounds[-1]

    return {
        "status": require_key(
            data,
            "rounds_completed",
            "FedAvg results",
        ),
        "partition_type": require_key(
            data,
            "partition_type",
            "FedAvg results",
        ),
        "seed": require_key(
            data,
            "seed",
            "FedAvg results",
        ),
        "client_count": len(
            require_key(
                data,
                "client_ids",
                "FedAvg results",
            )
        ),
        "local_epochs": require_key(
            data,
            "local_epochs",
            "FedAvg results",
        ),
        "batch_size": require_key(
            data,
            "batch_size",
            "FedAvg results",
        ),
        "learning_rate": require_key(
            data,
            "learning_rate",
            "FedAvg results",
        ),
        "device": require_key(
            data,
            "device",
            "FedAvg results",
        ),
        "parameter_count": require_key(
            data,
            "model_parameter_count",
            "FedAvg results",
        ),
        "total_train_rows": require_key(
            data,
            "total_training_rows",
            "FedAvg results",
        ),
        "total_validation_rows": require_key(
            data,
            "total_validation_rows",
            "FedAvg results",
        ),
        "rounds_requested": require_key(
            data,
            "rounds_requested",
            "FedAvg results",
        ),
        "rounds_completed": require_key(
            data,
            "rounds_completed",
            "FedAvg results",
        ),
        "final_round_metrics": {
            "round": last_round.get("round"),
            "validation_loss": last_round.get(
                "validation_loss"
            ),
            "validation_accuracy": last_round.get(
                "validation_accuracy"
            ),
            "validation_precision": last_round.get(
                "validation_precision"
            ),
            "validation_recall": last_round.get(
                "validation_recall"
            ),
            "validation_f1": last_round.get(
                "validation_f1"
            ),
            "validation_roc_auc": last_round.get(
                "validation_roc_auc"
            ),
        },
        "rounds": rounds,
        "total_elapsed_seconds": require_key(
            data,
            "total_experiment_wall_time_seconds",
            "FedAvg results",
        ),
        "resource_peak_rss_bytes": resource["measurement"][
            "peak_rss_bytes"
        ],
        "resource_peak_rss_mib": resource["measurement"][
            "peak_rss_mib"
        ],
        "resource_wall_time_seconds": resource[
            "wall_time_seconds"
        ],
    }


def extract_communication_summary(
    data: dict[str, Any],
) -> dict[str, Any]:
    """Extract communication-volume measurements."""

    measured_3_round = require_key(
        data,
        "measured_3_round_experiment",
        "communication measurement",
    )

    configured_10_round = require_key(
        data,
        "configured_10_round_experiment",
        "communication measurement",
    )

    return {
        "model_parameter_count": require_key(
            data,
            "model_parameter_count",
            "communication measurement",
        ),
        "model_state_dict_payload_bytes": require_key(
            data,
            "model_state_dict_payload_bytes",
            "communication measurement",
        ),
        "measured_rounds": require_key(
            data,
            "measured_rounds",
            "communication measurement",
        ),
        "measured_3_round_experiment": measured_3_round,
        "configured_10_round_experiment": configured_10_round,
        "measurement_method": require_key(
            data,
            "measurement_method",
            "communication measurement",
        ),
    }


def extract_resource_summary(
    resource: dict[str, Any],
) -> dict[str, Any]:
    """Extract the common resource measurement fields."""

    measurement = resource["measurement"]

    return {
        "status": resource["status"],
        "exit_code": resource["exit_code"],
        "peak_rss_bytes": measurement["peak_rss_bytes"],
        "peak_rss_mib": measurement["peak_rss_mib"],
        "sampling_interval_seconds": measurement[
            "sampling_interval_seconds"
        ],
        "samples_collected": measurement[
            "samples_collected"
        ],
        "wall_time_seconds": resource[
            "wall_time_seconds"
        ],
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    """Build the verified processed experiment summary."""

    # Load experiment results.
    centralized = load_json(CENTRALIZED_PATH)
    local_only = load_json(LOCAL_ONLY_PATH)
    device_non_iid = load_json(DEVICE_NON_IID_PATH)
    iid = load_json(IID_PATH)

    # Load supporting measurements.
    communication = load_json(COMMUNICATION_PATH)

    centralized_resource = load_json(
        CENTRALIZED_RESOURCE_PATH
    )
    local_only_resource = load_json(
        LOCAL_ONLY_RESOURCE_PATH
    )
    device_non_iid_resource = load_json(
        DEVICE_NON_IID_RESOURCE_PATH
    )
    iid_resource = load_json(
        IID_RESOURCE_PATH
    )

    # Validate source structures.
    validate_common_metadata(
        centralized=centralized,
        local_only=local_only,
        device_non_iid=device_non_iid,
        iid=iid,
        communication=communication,
    )

    validate_communication(communication)

    validate_resource_measurement(
        centralized_resource,
        "centralized resource measurement",
    )
    validate_resource_measurement(
        local_only_resource,
        "local-only resource measurement",
    )
    validate_resource_measurement(
        device_non_iid_resource,
        "device Non-IID resource measurement",
    )
    validate_resource_measurement(
        iid_resource,
        "IID resource measurement",
    )

    # Extract experiment summaries.
    centralized_summary = extract_centralized_summary(
        centralized,
        centralized_resource,
    )

    local_only_summary = extract_local_only_summary(
        local_only,
        local_only_resource,
    )

    device_non_iid_summary = extract_fedavg_summary(
        device_non_iid,
        device_non_iid_resource,
    )

    iid_summary = extract_fedavg_summary(
        iid,
        iid_resource,
    )

    communication_summary = extract_communication_summary(
        communication
    )

    # Extract resource-only summaries.
    resource_summary = {
        "centralized": extract_resource_summary(
            centralized_resource
        ),
        "local_only": extract_resource_summary(
            local_only_resource
        ),
        "device_non_iid_fedavg": extract_resource_summary(
            device_non_iid_resource
        ),
        "iid_fedavg": extract_resource_summary(
            iid_resource
        ),
    }

    # Cross-check training/validation row counts.
    require_equal(
        {
            "centralized.train": centralized_summary[
                "total_train_rows"
            ],
            "local_only.train": local_only_summary[
                "total_train_rows"
            ],
            "device_non_iid.train": device_non_iid_summary[
                "total_train_rows"
            ],
            "iid.train": iid_summary[
                "total_train_rows"
            ],
        },
        "total training rows",
    )

    require_equal(
        {
            "centralized.validation": centralized_summary[
                "total_validation_rows"
            ],
            "local_only.validation": local_only_summary[
                "total_validation_rows"
            ],
            "device_non_iid.validation": device_non_iid_summary[
                "total_validation_rows"
            ],
            "iid.validation": iid_summary[
                "total_validation_rows"
            ],
        },
        "total validation rows",
    )

    # Cross-check the communication model size.
    require_equal(
        {
            "centralized": centralized_summary[
                "parameter_count"
            ],
            "local_only": local_only_summary[
                "parameter_count"
            ],
            "device_non_iid": device_non_iid_summary[
                "parameter_count"
            ],
            "iid": iid_summary[
                "parameter_count"
            ],
            "communication": communication_summary[
                "model_parameter_count"
            ],
        },
        "model parameter count",
    )

    # The measured communication total must correspond to
    # the completed three-round experiments.
    measured_rounds = communication_summary[
        "measured_rounds"
    ]

    if measured_rounds != 3:
        raise ValueError(
            "Expected exactly 3 measured communication rounds, "
            f"got {measured_rounds!r}."
        )

    measured_total_bytes = communication_summary[
        "measured_3_round_experiment"
    ]["total_bytes"]

    if measured_total_bytes != 2059992:
        raise ValueError(
            "Unexpected measured 3-round communication total: "
            f"{measured_total_bytes!r} bytes."
        )

    # Build final processed summary.
    summary = {
        "project": "the study",
        "experiment_group": "N-BaIoT federated intrusion detection",
        "dataset": "N-BaIoT",
        "model": "small_mlp",
        "model_parameter_count": communication_summary[
            "model_parameter_count"
        ],
        "experiments": {
            "centralized": centralized_summary,
            "local_only": local_only_summary,
            "device_non_iid_fedavg": device_non_iid_summary,
            "iid_fedavg": iid_summary,
        },
        "communication": communication_summary,
        "resource_measurements": resource_summary,
        "methodological_notes": [
            "N-BaIoT devices are treated as simulated logical IoT clients.",
            "Federated experiments use FedAvg with full client participation.",
            "The persisted standard scaler was fitted using centralized training data and reused across clients.",
            "Therefore, the current experiments do not demonstrate a fully decentralized privacy-preserving preprocessing pipeline.",
            "No differential privacy or secure aggregation was implemented.",
            "Communication values represent model tensor payload accounting rather than measured network traffic.",
            "Communication accounting excludes protocol headers, serialization overhead, compression, encryption, and transport overhead.",
            "Resource measurements report sampled peak process-tree RSS and are not measurements of total system RAM or energy consumption.",
            "The centralized, device Non-IID FedAvg, and IID FedAvg results are not all evaluated under identical client-validation protocols; local-only models are evaluated on same-device validation data.",
            "These results use seed 42 and should be treated as a completed empirical run rather than evidence of generalization across random seeds.",
        ],
    }

    # Ensure output directory exists.
    RESULTS_PROCESSED.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Write deterministic, human-readable JSON.
    with OUTPUT_PATH.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            summary,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    # Final console verification.
    print("RESULTS SUMMARY BUILD: PASS")
    print(f"Output: {OUTPUT_PATH}")
    print(
        "Experiments summarized: "
        f"{len(summary['experiments'])}"
    )
    print(
        "Communication payload over 3 rounds: "
        f"{measured_total_bytes} bytes"
    )
    print(
        "Centralized elapsed seconds: "
        f"{centralized_summary['total_elapsed_seconds']}"
    )
    print(
        "Local-only elapsed seconds: "
        f"{local_only_summary['total_elapsed_seconds']}"
    )
    print(
        "Device Non-IID FedAvg elapsed seconds: "
        f"{device_non_iid_summary['total_elapsed_seconds']}"
    )
    print(
        "IID FedAvg elapsed seconds: "
        f"{iid_summary['total_elapsed_seconds']}"
    )
    print("All cross-experiment consistency checks passed.")


if __name__ == "__main__":
    main()

