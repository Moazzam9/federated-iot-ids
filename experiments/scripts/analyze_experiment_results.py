from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_RAW = PROJECT_ROOT / "results" / "raw"
RESULTS_PROCESSED = PROJECT_ROOT / "results" / "processed"
ANALYSIS_DIR = RESULTS_PROCESSED / "analysis"

CENTRALIZED_PATH = (
    RESULTS_RAW
    / "centralized"
    / "centralized_training_history.json"
)

LOCAL_ONLY_PATH = (
    RESULTS_RAW
    / "local_only"
    / "local_only_results.json"
)

DEVICE_NON_IID_PATH = (
    RESULTS_RAW
    / "device_non_iid_3round_seed42.json"
)

IID_PATH = (
    RESULTS_RAW
    / "iid_3round_seed42.json"
)

COMMUNICATION_PATH = (
    RESULTS_RAW
    / "communication"
    / "communication_measurement.json"
)

CENTRALIZED_RESOURCE_PATH = (
    RESULTS_RAW
    / "resource_measurement"
    / "centralized_3epoch.json"
)

LOCAL_ONLY_RESOURCE_PATH = (
    RESULTS_RAW
    / "resource_measurement"
    / "local_only_3epoch.json"
)

DEVICE_NON_IID_RESOURCE_PATH = (
    RESULTS_RAW
    / "resource_measurement"
    / "device_non_iid_3round_seed42.json"
)

IID_RESOURCE_PATH = (
    RESULTS_RAW
    / "resource_measurement"
    / "iid_3round_seed42.json"
)

OUTPUT_JSON = ANALYSIS_DIR / "experiment_analysis.json"

OUTPUT_EXPERIMENT_COMPARISON = (
    ANALYSIS_DIR / "experiment_comparison.csv"
)

OUTPUT_FEDAVG_ROUNDS = (
    ANALYSIS_DIR / "fedavg_round_metrics.csv"
)

OUTPUT_LOCAL_ONLY_DEVICES = (
    ANALYSIS_DIR / "local_only_device_metrics.csv"
)

OUTPUT_CLIENT_DISTRIBUTION = (
    ANALYSIS_DIR / "client_sample_distribution.csv"
)

OUTPUT_COMMUNICATION = (
    ANALYSIS_DIR / "communication_metrics.csv"
)

OUTPUT_RESOURCE = (
    ANALYSIS_DIR / "resource_metrics.csv"
)


# ----------------------------------------------------------------------
# General helpers
# ----------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.exists():
        raise FileNotFoundError(
            f"Required result file does not exist: {path}"
        )

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(
            f"Expected a JSON object in {path}, "
            f"got {type(data).__name__}."
        )

    return data


def require_key(
    data: dict[str, Any],
    key: str,
    source_name: str,
) -> Any:
    """Return a required key or raise a descriptive error."""
    if key not in data:
        raise KeyError(
            f"Missing required key {key!r} in {source_name}."
        )

    return data[key]


def write_csv(
    path: Path,
    fieldnames: list[str],
    rows: list[dict[str, Any]],
) -> None:
    """Write a list of dictionaries to CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
        )
        writer.writeheader()
        writer.writerows(rows)


def percentage_change(
    initial: float,
    final: float,
) -> float | None:
    """Calculate percentage change from initial to final."""
    if initial == 0:
        return None

    return ((final - initial) / initial) * 100.0


def absolute_difference(
    first: float,
    second: float,
) -> float:
    """Return first minus second."""
    return first - second


def round_metric(
    round_data: dict[str, Any],
    metric: str,
) -> float:
    """Read a validation metric from a FedAvg round."""
    validation = require_key(
        round_data,
        "validation",
        "FedAvg round",
    )

    if not isinstance(validation, dict):
        raise ValueError(
            "FedAvg round 'validation' must be an object."
        )

    value = require_key(
        validation,
        metric,
        "FedAvg round validation",
    )

    if not isinstance(value, (int, float)):
        raise TypeError(
            f"FedAvg metric {metric!r} must be numeric."
        )

    return float(value)


# ----------------------------------------------------------------------
# Source validation
# ----------------------------------------------------------------------


def validate_source_results(
    centralized: dict[str, Any],
    local_only: dict[str, Any],
    device_non_iid: dict[str, Any],
    iid: dict[str, Any],
    communication: dict[str, Any],
) -> None:
    """Validate the minimum expected structure before analysis."""

    require_key(
        centralized,
        "history",
        "centralized results",
    )

    require_key(
        local_only,
        "clients",
        "local-only results",
    )

    require_key(
        local_only,
        "macro_average",
        "local-only results",
    )

    require_key(
        local_only,
        "sample_weighted_average",
        "local-only results",
    )

    require_key(
        device_non_iid,
        "rounds",
        "device Non-IID results",
    )

    require_key(
        iid,
        "rounds",
        "IID results",
    )

    require_key(
        communication,
        "measurement_method",
        "communication results",
    )

    if not isinstance(
        centralized["history"],
        list,
    ):
        raise ValueError(
            "Centralized history must be a list."
        )

    if not isinstance(
        local_only["clients"],
        list,
    ):
        raise ValueError(
            "Local-only clients must be a list."
        )

    if not isinstance(
        device_non_iid["rounds"],
        list,
    ):
        raise ValueError(
            "Device Non-IID rounds must be a list."
        )

    if not isinstance(
        iid["rounds"],
        list,
    ):
        raise ValueError(
            "IID rounds must be a list."
        )

    if len(device_non_iid["rounds"]) != 3:
        raise ValueError(
            "Expected 3 device Non-IID rounds, "
            f"got {len(device_non_iid['rounds'])}."
        )

    if len(iid["rounds"]) != 3:
        raise ValueError(
            "Expected 3 IID rounds, "
            f"got {len(iid['rounds'])}."
        )


def validate_resource(
    resource: dict[str, Any],
    source_name: str,
) -> None:
    """Validate one resource measurement."""
    measurement = require_key(
        resource,
        "measurement",
        source_name,
    )

    if not isinstance(measurement, dict):
        raise ValueError(
            f"{source_name}.measurement must be an object."
        )

    for key in (
        "peak_rss_bytes",
        "peak_rss_mib",
        "sampling_interval_seconds",
        "samples_collected",
    ):
        require_key(
            measurement,
            key,
            f"{source_name}.measurement",
        )

    require_key(
        resource,
        "wall_time_seconds",
        source_name,
    )


# ----------------------------------------------------------------------
# Experiment comparison
# ----------------------------------------------------------------------


def build_experiment_comparison(
    centralized: dict[str, Any],
    local_only: dict[str, Any],
    device_non_iid: dict[str, Any],
    iid: dict[str, Any],
    resources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Build one comparable row per experimental condition."""

    centralized_history = centralized["history"]

    best_epoch = int(
        require_key(
            centralized,
            "best_epoch",
            "centralized results",
        )
    )

    best_epoch_record = None

    for record in centralized_history:
        if record["epoch"] == best_epoch:
            best_epoch_record = record
            break

    if best_epoch_record is None:
        raise ValueError(
            f"Could not find centralized epoch {best_epoch}."
        )

    local_macro = local_only["macro_average"]
    local_weighted = local_only[
        "sample_weighted_average"
    ]

    device_final = device_non_iid["rounds"][-1]["validation"]
    iid_final = iid["rounds"][-1]["validation"]

    rows = [
        {
            "experiment": "centralized",
            "evaluation_scope": "global_validation_best_epoch",
            "f1": best_epoch_record["validation_f1"],
            "roc_auc": best_epoch_record["validation_roc_auc"],
            "accuracy": best_epoch_record["validation_accuracy"],
            "precision": best_epoch_record["validation_precision"],
            "recall": best_epoch_record["validation_recall"],
            "loss": best_epoch_record["validation_loss"],
            "elapsed_seconds": centralized[
                "total_elapsed_seconds"
            ],
            "resource_peak_rss_mib": resources[
                "centralized"
            ]["measurement"]["peak_rss_mib"],
        },
        {
            "experiment": "local_only",
            "evaluation_scope": "same_device_validation_macro_average",
            "f1": local_macro["f1"],
            "roc_auc": local_macro["roc_auc"],
            "accuracy": local_macro["accuracy"],
            "precision": local_macro["precision"],
            "recall": local_macro["recall"],
            "loss": None,
            "elapsed_seconds": local_only[
                "total_experiment_wall_time_seconds"
            ],
            "resource_peak_rss_mib": resources[
                "local_only"
            ]["measurement"]["peak_rss_mib"],
        },
        {
            "experiment": "local_only_sample_weighted",
            "evaluation_scope": "same_device_validation_sample_weighted",
            "f1": local_weighted["f1"],
            "roc_auc": local_weighted["roc_auc"],
            "accuracy": local_weighted["accuracy"],
            "precision": local_weighted["precision"],
            "recall": local_weighted["recall"],
            "loss": None,
            "elapsed_seconds": local_only[
                "total_experiment_wall_time_seconds"
            ],
            "resource_peak_rss_mib": resources[
                "local_only"
            ]["measurement"]["peak_rss_mib"],
        },
        {
            "experiment": "device_non_iid_fedavg",
            "evaluation_scope": "global_validation_round_3",
            "f1": device_final["f1"],
            "roc_auc": device_final["roc_auc"],
            "accuracy": device_final["accuracy"],
            "precision": device_final["precision"],
            "recall": device_final["recall"],
            "loss": device_final["loss"],
            "elapsed_seconds": device_non_iid[
                "total_experiment_wall_time_seconds"
            ],
            "resource_peak_rss_mib": resources[
                "device_non_iid"
            ]["measurement"]["peak_rss_mib"],
        },
        {
            "experiment": "iid_fedavg",
            "evaluation_scope": "global_validation_round_3",
            "f1": iid_final["f1"],
            "roc_auc": iid_final["roc_auc"],
            "accuracy": iid_final["accuracy"],
            "precision": iid_final["precision"],
            "recall": iid_final["recall"],
            "loss": iid_final["loss"],
            "elapsed_seconds": iid[
                "total_experiment_wall_time_seconds"
            ],
            "resource_peak_rss_mib": resources[
                "iid"
            ]["measurement"]["peak_rss_mib"],
        },
    ]

    return rows


# ----------------------------------------------------------------------
# FedAvg round analysis
# ----------------------------------------------------------------------


def build_fedavg_round_metrics(
    device_non_iid: dict[str, Any],
    iid: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create a round-by-round table for both FedAvg conditions."""

    rows: list[dict[str, Any]] = []

    for experiment_name, experiment in (
        ("device_non_iid_fedavg", device_non_iid),
        ("iid_fedavg", iid),
    ):
        rounds = experiment["rounds"]

        for round_data in rounds:
            validation = round_data["validation"]

            rows.append(
                {
                    "experiment": experiment_name,
                    "round": round_data["round_number"],
                    "client_count": round_data["client_count"],
                    "total_client_samples": round_data[
                        "total_client_samples"
                    ],
                    "aggregation_elapsed_seconds": round_data[
                        "aggregation_elapsed_seconds"
                    ],
                    "measured_wall_time_seconds": round_data[
                        "measured_wall_time_seconds"
                    ],
                    "evaluation_time_seconds": validation[
                        "evaluation_time_seconds"
                    ],
                    "validation_loss": validation["loss"],
                    "validation_accuracy": validation[
                        "accuracy"
                    ],
                    "validation_precision": validation[
                        "precision"
                    ],
                    "validation_recall": validation[
                        "recall"
                    ],
                    "validation_f1": validation["f1"],
                    "validation_roc_auc": validation[
                        "roc_auc"
                    ],
                }
            )

    return rows


def build_fedavg_convergence_analysis(
    device_non_iid: dict[str, Any],
    iid: dict[str, Any],
) -> dict[str, Any]:
    """Calculate round-1 to round-3 convergence changes."""

    result: dict[str, Any] = {}

    metrics = (
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    )

    for name, experiment in (
        ("device_non_iid_fedavg", device_non_iid),
        ("iid_fedavg", iid),
    ):
        first = experiment["rounds"][0]["validation"]
        final = experiment["rounds"][-1]["validation"]

        result[name] = {
            "rounds": len(experiment["rounds"]),
            "round_1": {
                metric: first[metric]
                for metric in metrics
            },
            "round_3": {
                metric: final[metric]
                for metric in metrics
            },
            "absolute_change_round_1_to_round_3": {
                metric: final[metric] - first[metric]
                for metric in metrics
            },
            "percentage_change_round_1_to_round_3": {
                metric: percentage_change(
                    float(first[metric]),
                    float(final[metric]),
                )
                for metric in metrics
            },
        }

    return result


# ----------------------------------------------------------------------
# Local-only device analysis
# ----------------------------------------------------------------------


def build_local_only_device_metrics(
    local_only: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create a per-device local-only results table."""

    rows: list[dict[str, Any]] = []

    for client in local_only["clients"]:
        rows.append(
            {
                "client_id": client["client_id"],
                "training_samples": client[
                    "training_samples"
                ],
                "validation_samples": client[
                    "validation_samples"
                ],
                "epochs": client["epochs"],
                "training_loss": client[
                    "training_loss"
                ],
                "training_elapsed_seconds": client[
                    "training_elapsed_seconds"
                ],
                "validation_elapsed_seconds": client[
                    "validation_elapsed_seconds"
                ],
                "validation_loss": client[
                    "validation_loss"
                ],
                "accuracy": client["accuracy"],
                "precision": client["precision"],
                "recall": client["recall"],
                "f1": client["f1"],
                "roc_auc": client["roc_auc"],
            }
        )

    return rows


# ----------------------------------------------------------------------
# Client sample distribution
# ----------------------------------------------------------------------


def build_client_sample_distribution(
    local_only: dict[str, Any],
    device_non_iid: dict[str, Any],
    iid: dict[str, Any],
) -> list[dict[str, Any]]:
    """Describe client sample counts for local and FL experiments."""

    rows: list[dict[str, Any]] = []

    local_clients = local_only["clients"]

    local_total = sum(
        int(client["training_samples"])
        for client in local_clients
    )

    for client in local_clients:
        samples = int(client["training_samples"])

        rows.append(
            {
                "partition_type": "device_non_iid",
                "client_id": client["client_id"],
                "training_samples": samples,
                "training_sample_fraction": (
                    samples / local_total
                ),
            }
        )

    iid_clients = iid["rounds"][0]["clients"]

    iid_total = sum(
        int(client["samples"])
        for client in iid_clients
    )

    for client in iid_clients:
        samples = int(client["samples"])

        rows.append(
            {
                "partition_type": "iid",
                "client_id": client["client_id"],
                "training_samples": samples,
                "training_sample_fraction": (
                    samples / iid_total
                ),
            }
        )

    return rows


# ----------------------------------------------------------------------
# Communication analysis
# ----------------------------------------------------------------------


def build_communication_metrics(
    communication: dict[str, Any],
) -> list[dict[str, Any]]:
    """Create communication-volume analysis rows."""

    method = communication["measurement_method"]

    measured = communication[
        "measured_3_round_experiment"
    ]

    projected = communication[
        "configured_10_round_experiment"
    ]

    per_client = communication[
        "per_client_per_round"
    ]

    per_round = communication[
        "per_round_all_clients"
    ]

    return [
        {
            "metric": "model_parameter_count",
            "value": communication[
                "model_parameter_count"
            ],
            "unit": "parameters",
            "scope": "model",
        },
        {
            "metric": "model_state_dict_payload",
            "value": communication[
                "model_state_dict_payload_bytes"
            ],
            "unit": "bytes",
            "scope": "one_model",
        },
        {
            "metric": "per_client_download",
            "value": per_client[
                "download_bytes"
            ],
            "unit": "bytes",
            "scope": "per_client_per_round",
        },
        {
            "metric": "per_client_upload",
            "value": per_client[
                "upload_bytes"
            ],
            "unit": "bytes",
            "scope": "per_client_per_round",
        },
        {
            "metric": "all_clients_total_per_round",
            "value": per_round[
                "total_bytes"
            ],
            "unit": "bytes",
            "scope": "all_clients_one_round",
        },
        {
            "metric": "measured_3_round_total",
            "value": measured["total_bytes"],
            "unit": "bytes",
            "scope": "completed_three_round_experiments",
        },
        {
            "metric": "projected_10_round_total",
            "value": projected["total_bytes"],
            "unit": "bytes",
            "scope": "projection",
        },
        {
            "metric": "actual_network_transfer",
            "value": method[
                "actual_network_transfer"
            ],
            "unit": "boolean",
            "scope": "measurement_method",
        },
        {
            "metric": "network_traffic_measured",
            "value": method[
                "network_traffic_measured"
            ],
            "unit": "boolean",
            "scope": "measurement_method",
        },
    ]


# ----------------------------------------------------------------------
# Resource analysis
# ----------------------------------------------------------------------


def build_resource_metrics(
    resources: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Create a resource-measurement comparison table."""

    rows: list[dict[str, Any]] = []

    for experiment_name, resource in resources.items():
        measurement = resource["measurement"]

        rows.append(
            {
                "experiment": experiment_name,
                "status": resource["status"],
                "exit_code": resource["exit_code"],
                "peak_rss_bytes": measurement[
                    "peak_rss_bytes"
                ],
                "peak_rss_mib": measurement[
                    "peak_rss_mib"
                ],
                "sampling_interval_seconds": measurement[
                    "sampling_interval_seconds"
                ],
                "samples_collected": measurement[
                    "samples_collected"
                ],
                "monitor_wall_time_seconds": resource[
                    "wall_time_seconds"
                ],
            }
        )

    return rows


# ----------------------------------------------------------------------
# Derived comparisons
# ----------------------------------------------------------------------


def build_derived_comparisons(
    centralized: dict[str, Any],
    local_only: dict[str, Any],
    device_non_iid: dict[str, Any],
    iid: dict[str, Any],
    resources: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    """Calculate explicitly defined differences without ranking."""

    device_final = device_non_iid["rounds"][-1]["validation"]
    iid_final = iid["rounds"][-1]["validation"]

    centralized_history = centralized["history"]

    best_epoch = int(centralized["best_epoch"])

    centralized_best = next(
        record
        for record in centralized_history
        if record["epoch"] == best_epoch
    )

    local_macro = local_only["macro_average"]
    local_weighted = local_only[
        "sample_weighted_average"
    ]

    return {
        "final_round_iid_minus_device_non_iid": {
            "f1": absolute_difference(
                iid_final["f1"],
                device_final["f1"],
            ),
            "roc_auc": absolute_difference(
                iid_final["roc_auc"],
                device_final["roc_auc"],
            ),
            "accuracy": absolute_difference(
                iid_final["accuracy"],
                device_final["accuracy"],
            ),
            "precision": absolute_difference(
                iid_final["precision"],
                device_final["precision"],
            ),
            "recall": absolute_difference(
                iid_final["recall"],
                device_final["recall"],
            ),
            "loss": absolute_difference(
                iid_final["loss"],
                device_final["loss"],
            ),
        },
        "local_only_macro_minus_sample_weighted": {
            "f1": absolute_difference(
                local_macro["f1"],
                local_weighted["f1"],
            ),
            "roc_auc": absolute_difference(
                local_macro["roc_auc"],
                local_weighted["roc_auc"],
            ),
            "accuracy": absolute_difference(
                local_macro["accuracy"],
                local_weighted["accuracy"],
            ),
            "precision": absolute_difference(
                local_macro["precision"],
                local_weighted["precision"],
            ),
            "recall": absolute_difference(
                local_macro["recall"],
                local_weighted["recall"],
            ),
        },
        "runtime_seconds": {
            "centralized": centralized[
                "total_elapsed_seconds"
            ],
            "local_only": local_only[
                "total_experiment_wall_time_seconds"
            ],
            "device_non_iid_fedavg": device_non_iid[
                "total_experiment_wall_time_seconds"
            ],
            "iid_fedavg": iid[
                "total_experiment_wall_time_seconds"
            ],
        },
        "peak_rss_mib": {
            "centralized": resources[
                "centralized"
            ]["measurement"]["peak_rss_mib"],
            "local_only": resources[
                "local_only"
            ]["measurement"]["peak_rss_mib"],
            "device_non_iid_fedavg": resources[
                "device_non_iid"
            ]["measurement"]["peak_rss_mib"],
            "iid_fedavg": resources[
                "iid"
            ]["measurement"]["peak_rss_mib"],
        },
        "centralized_best_validation": {
            "epoch": best_epoch,
            "f1": centralized_best[
                "validation_f1"
            ],
            "roc_auc": centralized_best[
                "validation_roc_auc"
            ],
            "accuracy": centralized_best[
                "validation_accuracy"
            ],
            "precision": centralized_best[
                "validation_precision"
            ],
            "recall": centralized_best[
                "validation_recall"
            ],
            "loss": centralized_best[
                "validation_loss"
            ],
        },
    }


# ----------------------------------------------------------------------
# Main analysis
# ----------------------------------------------------------------------


def main() -> None:
    """Run the complete reproducible results analysis."""

    print("Loading verified experiment results...")

    centralized = load_json(
        CENTRALIZED_PATH
    )

    local_only = load_json(
        LOCAL_ONLY_PATH
    )

    device_non_iid = load_json(
        DEVICE_NON_IID_PATH
    )

    iid = load_json(
        IID_PATH
    )

    communication = load_json(
        COMMUNICATION_PATH
    )

    resources = {
        "centralized": load_json(
            CENTRALIZED_RESOURCE_PATH
        ),
        "local_only": load_json(
            LOCAL_ONLY_RESOURCE_PATH
        ),
        "device_non_iid": load_json(
            DEVICE_NON_IID_RESOURCE_PATH
        ),
        "iid": load_json(
            IID_RESOURCE_PATH
        ),
    }

    validate_source_results(
        centralized=centralized,
        local_only=local_only,
        device_non_iid=device_non_iid,
        iid=iid,
        communication=communication,
    )

    for name, resource in resources.items():
        validate_resource(
            resource,
            f"{name} resource measurement",
        )

    print("Source validation: PASS")

    # Build analysis sections.
    experiment_comparison = build_experiment_comparison(
        centralized=centralized,
        local_only=local_only,
        device_non_iid=device_non_iid,
        iid=iid,
        resources=resources,
    )

    fedavg_round_metrics = build_fedavg_round_metrics(
        device_non_iid=device_non_iid,
        iid=iid,
    )

    convergence = build_fedavg_convergence_analysis(
        device_non_iid=device_non_iid,
        iid=iid,
    )

    local_only_devices = build_local_only_device_metrics(
        local_only
    )

    client_distribution = (
        build_client_sample_distribution(
            local_only=local_only,
            device_non_iid=device_non_iid,
            iid=iid,
        )
    )

    communication_metrics = (
        build_communication_metrics(
            communication
        )
    )

    resource_metrics = build_resource_metrics(
        resources
    )

    derived_comparisons = build_derived_comparisons(
        centralized=centralized,
        local_only=local_only,
        device_non_iid=device_non_iid,
        iid=iid,
        resources=resources,
    )

    # ------------------------------------------------------------------
    # Write CSV outputs.
    # ------------------------------------------------------------------

    write_csv(
        OUTPUT_EXPERIMENT_COMPARISON,
        [
            "experiment",
            "evaluation_scope",
            "f1",
            "roc_auc",
            "accuracy",
            "precision",
            "recall",
            "loss",
            "elapsed_seconds",
            "resource_peak_rss_mib",
        ],
        experiment_comparison,
    )

    write_csv(
        OUTPUT_FEDAVG_ROUNDS,
        [
            "experiment",
            "round",
            "client_count",
            "total_client_samples",
            "aggregation_elapsed_seconds",
            "measured_wall_time_seconds",
            "evaluation_time_seconds",
            "validation_loss",
            "validation_accuracy",
            "validation_precision",
            "validation_recall",
            "validation_f1",
            "validation_roc_auc",
        ],
        fedavg_round_metrics,
    )

    write_csv(
        OUTPUT_LOCAL_ONLY_DEVICES,
        [
            "client_id",
            "training_samples",
            "validation_samples",
            "epochs",
            "training_loss",
            "training_elapsed_seconds",
            "validation_elapsed_seconds",
            "validation_loss",
            "accuracy",
            "precision",
            "recall",
            "f1",
            "roc_auc",
        ],
        local_only_devices,
    )

    write_csv(
        OUTPUT_CLIENT_DISTRIBUTION,
        [
            "partition_type",
            "client_id",
            "training_samples",
            "training_sample_fraction",
        ],
        client_distribution,
    )

    write_csv(
        OUTPUT_COMMUNICATION,
        [
            "metric",
            "value",
            "unit",
            "scope",
        ],
        communication_metrics,
    )

    write_csv(
        OUTPUT_RESOURCE,
        [
            "experiment",
            "status",
            "exit_code",
            "peak_rss_bytes",
            "peak_rss_mib",
            "sampling_interval_seconds",
            "samples_collected",
            "monitor_wall_time_seconds",
        ],
        resource_metrics,
    )

    # ------------------------------------------------------------------
    # Build JSON analysis document.
    # ------------------------------------------------------------------

    analysis = {
        "project": "Publication 02",
        "experiment_group": (
            "N-BaIoT federated intrusion detection"
        ),
        "dataset": "N-BaIoT",
        "model": "small_mlp",
        "seed": 42,
        "analysis_scope": {
            "centralized": (
                "best validation epoch from 3-epoch run"
            ),
            "local_only": (
                "same-device validation; macro and "
                "sample-weighted summaries"
            ),
            "device_non_iid_fedavg": (
                "global validation across three FedAvg rounds"
            ),
            "iid_fedavg": (
                "global validation across three FedAvg rounds"
            ),
        },
        "experiment_comparison": experiment_comparison,
        "fedavg_round_analysis": convergence,
        "local_only_device_metrics": local_only_devices,
        "client_sample_distribution": client_distribution,
        "communication_metrics": communication_metrics,
        "resource_metrics": resource_metrics,
        "derived_comparisons": derived_comparisons,
        "interpretation_boundaries": [
            "The analysis reports measured results and explicitly defined differences; it does not assign an overall winner or ranking.",
            "Local-only models use same-device validation, whereas centralized and FedAvg models use the global validation split, so direct performance comparison has different evaluation scopes.",
            "The experiments use seed 42 only; variability across random seeds has not been measured.",
            "The persisted scaler was fitted using centralized training data and reused by federated clients.",
            "Federated learning therefore does not represent a fully decentralized preprocessing pipeline in this experiment.",
            "No differential privacy or secure aggregation was implemented.",
            "Clients represent simulated logical IoT participants derived from N-BaIoT devices.",
            "Communication results are model tensor payload calculations, not measurements of actual network traffic.",
            "Communication calculations exclude protocol headers, serialization, compression, encryption, and transport overhead.",
            "Peak RSS is sampled process-tree memory usage, not energy consumption or a hardware-level instantaneous memory maximum.",
            "A single measured run is insufficient to establish statistical generalization across seeds or datasets.",
        ],
    }

    ANALYSIS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_JSON.open(
        "w",
        encoding="utf-8",
        newline="\n",
    ) as handle:
        json.dump(
            analysis,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")

    # ------------------------------------------------------------------
    # Console summary.
    # ------------------------------------------------------------------

    print()
    print("EXPERIMENT RESULTS ANALYSIS: PASS")
    print(
        f"Output directory: {ANALYSIS_DIR}"
    )
    print(
        f"Experiment comparison rows: "
        f"{len(experiment_comparison)}"
    )
    print(
        f"FedAvg round rows: "
        f"{len(fedavg_round_metrics)}"
    )
    print(
        f"Local-only device rows: "
        f"{len(local_only_devices)}"
    )
    print(
        f"Client distribution rows: "
        f"{len(client_distribution)}"
    )
    print(
        f"Communication rows: "
        f"{len(communication_metrics)}"
    )
    print(
        f"Resource rows: "
        f"{len(resource_metrics)}"
    )

    print()
    print("Round-1 to Round-3 F1 changes:")

    for experiment_name, values in convergence.items():
        change = values[
            "absolute_change_round_1_to_round_3"
        ]["f1"]

        print(
            f"  {experiment_name}: "
            f"{change:+.12f}"
        )

    print()
    print("Round-1 to Round-3 ROC-AUC changes:")

    for experiment_name, values in convergence.items():
        change = values[
            "absolute_change_round_1_to_round_3"
        ]["roc_auc"]

        print(
            f"  {experiment_name}: "
            f"{change:+.12f}"
        )

    print()
    print(
        "Analysis artifacts written successfully."
    )


if __name__ == "__main__":
    main()