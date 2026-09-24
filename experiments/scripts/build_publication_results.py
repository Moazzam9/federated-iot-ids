from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import pandas as pd


# ----------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RESULTS_RAW = PROJECT_ROOT / "results" / "raw"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "processed" / "publication"

TABLES_ROOT = OUTPUT_ROOT / "tables"
FIGURES_ROOT = OUTPUT_ROOT / "figures"

CENTRALIZED_RESULT = (
    RESULTS_RAW / "centralized" / "centralized_training_history.json"
)
LOCAL_ONLY_RESULT = RESULTS_RAW / "local_only" / "local_only_results.json"
DEVICE_NON_IID_RESULT = (
    RESULTS_RAW / "device_non_iid_3round_seed42.json"
)
IID_RESULT = RESULTS_RAW / "iid_3round_seed42.json"

COMMUNICATION_RESULT = (
    RESULTS_RAW / "communication" / "communication_measurement.json"
)

RESOURCE_ROOT = RESULTS_RAW / "resource_measurement"

FINAL_TEST_RESULT = (
    RESULTS_RAW
    / "final_test_evaluation"
    / "final_test_evaluation.json"
)

VALIDATION_TO_TEST_CSV = (
    PROJECT_ROOT
    / "results"
    / "processed"
    / "analysis"
    / "validation_to_test_comparison.csv"
)


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------


def load_json(path: Path) -> dict[str, Any]:
    """Load a JSON object from disk."""
    if not path.exists():
        raise FileNotFoundError(f"Required JSON file does not exist: {path}")

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object at {path}")

    return data


def require_keys(
    data: dict[str, Any],
    required: list[str],
    source_name: str,
) -> None:
    """Raise a clear error when required keys are absent."""
    missing = [key for key in required if key not in data]

    if missing:
        raise ValueError(
            f"{source_name} is missing required keys: {missing}"
        )


def require_file(path: Path, description: str) -> None:
    """Verify that a required file exists."""
    if not path.exists():
        raise FileNotFoundError(
            f"{description} does not exist: {path}"
        )


def write_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write rows to a CSV file."""
    if not rows:
        raise ValueError(f"Cannot write empty table: {path}")

    path.parent.mkdir(parents=True, exist_ok=True)
    dataframe = pd.DataFrame(rows)
    dataframe.to_csv(path, index=False)


def write_json(data: dict[str, Any], path: Path) -> None:
    """Write JSON with stable formatting."""
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as handle:
        json.dump(
            data,
            handle,
            indent=2,
            ensure_ascii=False,
        )
        handle.write("\n")


def save_figure(path: Path) -> None:
    """Save the current matplotlib figure."""
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=300, bbox_inches="tight")
    plt.close()


# ----------------------------------------------------------------------
# Source validation
# ----------------------------------------------------------------------


def validate_source_results(
    centralized: dict[str, Any],
    local_only: dict[str, Any],
    device_non_iid: dict[str, Any],
    iid: dict[str, Any],
    communication: dict[str, Any],
    final_test: dict[str, Any],
) -> None:
    """Validate the verified result structures before publication packaging."""

    require_keys(
        centralized,
        [
            "experiment",
            "dataset",
            "model",
            "seed",
            "epochs",
            "batch_size",
            "parameter_count",
            "history",
        ],
        "centralized result",
    )

    require_keys(
        local_only,
        [
            "experiment",
            "status",
            "dataset",
            "model",
            "partition_type",
            "seed",
            "client_count",
            "client_ids",
            "epochs_per_client",
            "batch_size",
            "learning_rate",
            "device",
            "model_parameter_count",
            "total_training_rows",
            "total_validation_rows",
            "client_training_sample_counts",
            "clients",
            "macro_average",
            "sample_weighted_average",
            "total_experiment_wall_time_seconds",
        ],
        "local-only result",
    )

    for name, result in [
        ("device Non-IID FedAvg", device_non_iid),
        ("IID FedAvg", iid),
    ]:
        require_keys(
            result,
            [
                "experiment",
                "status",
                "partition_type",
                "seed",
                "clients",
                "client_ids",
                "local_epochs",
                "batch_size",
                "learning_rate",
                "device",
                "model_parameter_count",
                "rounds_requested",
                "rounds_completed",
                "total_training_rows",
                "total_validation_rows",
                "client_sample_counts",
                "rounds",
                "total_experiment_wall_time_seconds",
            ],
            f"{name} result",
        )

        if result["rounds_completed"] != len(result["rounds"]):
            raise ValueError(
                f"{name}: rounds_completed does not match rounds length."
            )

        for round_result in result["rounds"]:
            require_keys(
                round_result,
                [
                    "round_number",
                    "client_count",
                    "total_client_samples",
                    "aggregation_elapsed_seconds",
                    "measured_wall_time_seconds",
                    "validation",
                    "clients",
                ],
                f"{name} round",
            )

            require_keys(
                round_result["validation"],
                [
                    "loss",
                    "samples",
                    "accuracy",
                    "precision",
                    "recall",
                    "f1",
                    "roc_auc",
                    "evaluation_time_seconds",
                ],
                f"{name} round validation",
            )

    require_keys(
        communication,
        [
            "experiment",
            "status",
            "model",
            "client_count",
            "measured_rounds",
            "configured_max_federated_rounds",
            "model_parameter_count",
            "model_state_dict_payload_bytes",
            "per_client_per_round",
            "per_round_all_clients",
            "measured_3_round_experiment",
            "configured_10_round_experiment",
            "measurement_method",
        ],
        "communication result",
    )

    require_keys(
        communication["measurement_method"],
        [
            "description",
            "actual_network_transfer",
            "network_traffic_measured",
            "protocol_headers_included",
            "serialization_overhead_included",
            "compression_included",
            "encryption_overhead_included",
            "transport_overhead_included",
            "clients_per_round",
            "full_client_participation",
        ],
        "communication measurement_method",
    )

    require_keys(
        communication["measured_3_round_experiment"],
        [
            "download_bytes",
            "upload_bytes",
            "total_bytes",
            "download_mib",
            "upload_mib",
            "total_mib",
            "download_mb",
            "upload_mb",
            "total_mb",
        ],
        "communication measured_3_round_experiment",
    )

    require_keys(
        communication["configured_10_round_experiment"],
        [
            "download_bytes",
            "upload_bytes",
            "total_bytes",
            "download_mib",
            "upload_mib",
            "total_mib",
            "download_mb",
            "upload_mb",
            "total_mb",
        ],
        "communication configured_10_round_experiment",
    )

    require_keys(
        final_test,
        [
            "experiment",
            "status",
            "dataset",
            "model",
            "seed",
            "device",
            "batch_size",
            "model_parameter_count",
            "frozen_global_test_rows",
            "local_only_test_rows_sum",
            "scaler",
            "centralized",
            "device_non_iid_fedavg",
            "iid_fedavg",
            "local_only",
        ],
        "final-test result",
    )

    for name in [
        "centralized",
        "device_non_iid_fedavg",
        "iid_fedavg",
    ]:
        require_keys(
            final_test[name],
            [
                "checkpoint_path",
                "evaluation_scope",
                "samples",
                "loss",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "evaluation_time_seconds",
            ],
            f"final-test {name}",
        )

    require_keys(
        final_test["local_only"],
        [
            "checkpoint_root",
            "evaluation_scope",
            "client_count",
            "clients",
            "macro_average",
            "sample_weighted_average",
        ],
        "final-test local_only",
    )

    if len(final_test["local_only"]["clients"]) != final_test["local_only"]["client_count"]:
        raise ValueError(
            "Final-test local-only client count does not match "
            "the number of client records."
        )


# ----------------------------------------------------------------------
# Table builders
# ----------------------------------------------------------------------


def build_main_performance(
    final_test: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build the main final-test performance table."""

    rows: list[dict[str, Any]] = []

    global_experiments = [
        (
            "centralized",
            "Centralized",
            final_test["centralized"],
        ),
        (
            "device_non_iid_fedavg",
            "Device Non-IID FedAvg",
            final_test["device_non_iid_fedavg"],
        ),
        (
            "iid_fedavg",
            "IID FedAvg",
            final_test["iid_fedavg"],
        ),
    ]

    for experiment_id, label, result in global_experiments:
        rows.append(
            {
                "experiment": experiment_id,
                "label": label,
                "test_scope": result["evaluation_scope"],
                "test_rows": result["samples"],
                "loss": result["loss"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "roc_auc": result["roc_auc"],
                "evaluation_time_seconds": result[
                    "evaluation_time_seconds"
                ],
            }
        )

    local_only = final_test["local_only"]

    rows.append(
        {
            "experiment": "local_only",
            "label": "Local-only (macro)",
            "test_scope": local_only["evaluation_scope"],
            "test_rows": sum(
                client["samples"]
                for client in local_only["clients"]
            ),
            "loss": local_only["macro_average"]["loss"],
            "accuracy": local_only["macro_average"]["accuracy"],
            "precision": local_only["macro_average"]["precision"],
            "recall": local_only["macro_average"]["recall"],
            "f1": local_only["macro_average"]["f1"],
            "roc_auc": local_only["macro_average"]["roc_auc"],
            "evaluation_time_seconds": sum(
                client["evaluation_time_seconds"]
                for client in local_only["clients"]
            ),
        }
    )

    rows.append(
        {
            "experiment": "local_only",
            "label": "Local-only (sample-weighted)",
            "test_scope": local_only["evaluation_scope"],
            "test_rows": sum(
                client["samples"]
                for client in local_only["clients"]
            ),
            "loss": local_only["sample_weighted_average"]["loss"],
            "accuracy": local_only["sample_weighted_average"][
                "accuracy"
            ],
            "precision": local_only["sample_weighted_average"][
                "precision"
            ],
            "recall": local_only["sample_weighted_average"][
                "recall"
            ],
            "f1": local_only["sample_weighted_average"]["f1"],
            "roc_auc": local_only["sample_weighted_average"][
                "roc_auc"
            ],
            "evaluation_time_seconds": sum(
                client["evaluation_time_seconds"]
                for client in local_only["clients"]
            ),
        }
    )

    return rows


def build_fedavg_convergence(
    device_non_iid: dict[str, Any],
    iid: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build FedAvg round-by-round validation metrics."""

    rows: list[dict[str, Any]] = []

    for experiment_id, result in [
        ("device_non_iid_fedavg", device_non_iid),
        ("iid_fedavg", iid),
    ]:
        for round_result in result["rounds"]:
            validation = round_result["validation"]

            rows.append(
                {
                    "experiment": experiment_id,
                    "round": round_result["round_number"],
                    "client_count": round_result["client_count"],
                    "total_client_samples": round_result[
                        "total_client_samples"
                    ],
                    "aggregation_elapsed_seconds": round_result[
                        "aggregation_elapsed_seconds"
                    ],
                    "measured_wall_time_seconds": round_result[
                        "measured_wall_time_seconds"
                    ],
                    "validation_loss": validation["loss"],
                    "validation_accuracy": validation["accuracy"],
                    "validation_precision": validation["precision"],
                    "validation_recall": validation["recall"],
                    "validation_f1": validation["f1"],
                    "validation_roc_auc": validation["roc_auc"],
                    "evaluation_time_seconds": validation[
                        "evaluation_time_seconds"
                    ],
                }
            )

    return rows


def build_local_only_by_device(
    final_test: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build local-only same-device final-test metrics."""

    rows: list[dict[str, Any]] = []

    for client in final_test["local_only"]["clients"]:
        rows.append(
            {
                "client_id": client["client_id"],
                "test_scope": final_test["local_only"][
                    "evaluation_scope"
                ],
                "test_rows": client["samples"],
                "loss": client["loss"],
                "accuracy": client["accuracy"],
                "precision": client["precision"],
                "recall": client["recall"],
                "f1": client["f1"],
                "roc_auc": client["roc_auc"],
                "evaluation_time_seconds": client[
                    "evaluation_time_seconds"
                ],
            }
        )

    return rows


def build_communication_cost(
    communication: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build publication-ready communication-volume table."""

    method = communication["measurement_method"]
    per_client = communication["per_client_per_round"]
    per_round = communication["per_round_all_clients"]
    measured = communication["measured_3_round_experiment"]
    configured = communication["configured_10_round_experiment"]

    rows = [
        {
            "scenario": "per_client_per_round",
            "rounds": 1,
            "clients_per_round": method["clients_per_round"],
            "download_bytes": per_client["download_bytes"],
            "upload_bytes": per_client["upload_bytes"],
            "total_bytes": per_client["total_bytes"],
            "total_mib": (
                per_client["total_bytes"]
                / (1024 * 1024)
            ),
            "total_mb_decimal": (
                per_client["total_bytes"] / 1_000_000
            ),
            "status": "per-round estimate",
        },
        {
            "scenario": "all_clients_per_round",
            "rounds": 1,
            "clients_per_round": method["clients_per_round"],
            "download_bytes": per_round["download_bytes"],
            "upload_bytes": per_round["upload_bytes"],
            "total_bytes": per_round["total_bytes"],
            "total_mib": per_round["total_mib"],
            "total_mb_decimal": (
                per_round["total_bytes"] / 1_000_000
            ),
            "status": "per-round estimate",
        },
        {
            "scenario": "measured_3_round_experiment",
            "rounds": communication["measured_rounds"],
            "clients_per_round": method["clients_per_round"],
            "download_bytes": measured["download_bytes"],
            "upload_bytes": measured["upload_bytes"],
            "total_bytes": measured["total_bytes"],
            "total_mib": measured["total_mib"],
            "total_mb_decimal": measured["total_mb"],
            "status": "completed experiments",
        },
        {
            "scenario": "configured_10_round_projection",
            "rounds": communication[
                "configured_max_federated_rounds"
            ],
            "clients_per_round": method["clients_per_round"],
            "download_bytes": configured["download_bytes"],
            "upload_bytes": configured["upload_bytes"],
            "total_bytes": configured["total_bytes"],
            "total_mib": configured["total_mib"],
            "total_mb_decimal": configured["total_mb"],
            "status": "projection",
        },
    ]

    return rows


def load_resource_metrics() -> list[dict[str, Any]]:
    """Load the four verified resource measurements."""

    mapping = {
        "centralized": RESOURCE_ROOT / "centralized_3epoch.json",
        "local_only": RESOURCE_ROOT / "local_only_3epoch.json",
        "device_non_iid_fedavg": (
            RESOURCE_ROOT
            / "device_non_iid_3round_seed42.json"
        ),
        "iid_fedavg": RESOURCE_ROOT / "iid_3round_seed42.json",
    }

    rows: list[dict[str, Any]] = []

    for experiment, path in mapping.items():
        data = load_json(path)

        require_keys(
            data,
            [
                "status",
                "exit_code",
                "measurement",
                "wall_time_seconds",
            ],
            f"resource measurement {experiment}",
        )

        require_keys(
            data["measurement"],
            [
                "metric",
                "peak_rss_bytes",
                "peak_rss_mib",
                "sampling_interval_seconds",
                "samples_collected",
            ],
            f"resource measurement {experiment} measurement",
        )

        rows.append(
            {
                "experiment": experiment,
                "status": data["status"],
                "exit_code": data["exit_code"],
                "metric": data["measurement"]["metric"],
                "peak_rss_bytes": data["measurement"][
                    "peak_rss_bytes"
                ],
                "peak_rss_mib": data["measurement"]["peak_rss_mib"],
                "sampling_interval_seconds": data["measurement"][
                    "sampling_interval_seconds"
                ],
                "samples_collected": data["measurement"][
                    "samples_collected"
                ],
                "wall_time_seconds": data["wall_time_seconds"],
            }
        )

    return rows


def load_validation_to_test_table() -> list[dict[str, Any]]:
    """Load the existing verified validation-to-test comparison."""

    require_file(
        VALIDATION_TO_TEST_CSV,
        "Validation-to-test comparison CSV",
    )

    dataframe = pd.read_csv(VALIDATION_TO_TEST_CSV)

    if dataframe.empty:
        raise ValueError(
            "Validation-to-test comparison CSV is empty."
        )

    return dataframe.to_dict(orient="records")


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------


def build_fedavg_convergence_figures(
    convergence_rows: list[dict[str, Any]],
) -> None:
    """Create FedAvg F1 and ROC-AUC convergence figures."""

    dataframe = pd.DataFrame(convergence_rows)

    for metric, ylabel, filename in [
        (
            "validation_f1",
            "Validation F1",
            "fedavg_f1_convergence.png",
        ),
        (
            "validation_roc_auc",
            "Validation ROC-AUC",
            "fedavg_roc_auc_convergence.png",
        ),
    ]:
        plt.figure(figsize=(8, 5))

        for experiment_id in [
            "device_non_iid_fedavg",
            "iid_fedavg",
        ]:
            subset = dataframe[
                dataframe["experiment"] == experiment_id
            ].sort_values("round")

            plt.plot(
                subset["round"],
                subset[metric],
                marker="o",
                label=experiment_id,
            )

        plt.xlabel("Federated round")
        plt.ylabel(ylabel)
        plt.xticks(sorted(dataframe["round"].unique()))
        plt.grid(True, alpha=0.3)
        plt.legend()

        save_figure(FIGURES_ROOT / filename)


def build_local_only_f1_figure(
    local_rows: list[dict[str, Any]],
) -> None:
    """Create local-only F1 by device."""

    dataframe = (
        pd.DataFrame(local_rows)
        .sort_values("f1", ascending=False)
    )

    plt.figure(figsize=(11, 6))

    plt.bar(
        dataframe["client_id"],
        dataframe["f1"],
    )

    plt.ylabel("Test F1")
    plt.xlabel("Simulated IoT client / device")
    plt.xticks(rotation=60, ha="right")
    plt.ylim(
        max(0.0, float(dataframe["f1"].min()) - 0.03),
        1.0,
    )
    plt.grid(axis="y", alpha=0.3)

    save_figure(
        FIGURES_ROOT / "local_only_f1_by_device.png"
    )


def build_runtime_figure(
    resource_rows: list[dict[str, Any]],
) -> None:
    """Create experiment wall-time comparison."""

    dataframe = pd.DataFrame(resource_rows)

    plt.figure(figsize=(9, 5))

    plt.bar(
        dataframe["experiment"],
        dataframe["wall_time_seconds"] / 60.0,
    )

    plt.ylabel("Measured wall time (minutes)")
    plt.xlabel("Experiment")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    save_figure(FIGURES_ROOT / "experiment_runtime.png")


def build_memory_figure(
    resource_rows: list[dict[str, Any]],
) -> None:
    """Create peak memory comparison."""

    dataframe = pd.DataFrame(resource_rows)

    plt.figure(figsize=(9, 5))

    plt.bar(
        dataframe["experiment"],
        dataframe["peak_rss_mib"],
    )

    plt.ylabel("Peak process-tree RSS (MiB)")
    plt.xlabel("Experiment")
    plt.xticks(rotation=25, ha="right")
    plt.grid(axis="y", alpha=0.3)

    save_figure(FIGURES_ROOT / "peak_memory_usage.png")


def build_communication_figure(
    communication_rows: list[dict[str, Any]],
) -> None:
    """Create communication payload comparison."""

    dataframe = pd.DataFrame(
        [
            row
            for row in communication_rows
            if row["scenario"]
            in [
                "all_clients_per_round",
                "measured_3_round_experiment",
                "configured_10_round_projection",
            ]
        ]
    )

    plt.figure(figsize=(9, 5))

    labels = [
        "Per round",
        "3 rounds",
        "10-round projection",
    ]

    values = dataframe["total_mb_decimal"].tolist()

    plt.bar(labels, values)

    plt.ylabel("Total model payload (MB, decimal)")
    plt.xlabel("Communication scenario")
    plt.grid(axis="y", alpha=0.3)

    save_figure(
        FIGURES_ROOT / "communication_payload.png"
    )


# ----------------------------------------------------------------------
# Analysis JSON
# ----------------------------------------------------------------------


def build_publication_analysis(
    centralized: dict[str, Any],
    local_only: dict[str, Any],
    device_non_iid: dict[str, Any],
    iid: dict[str, Any],
    communication: dict[str, Any],
    final_test: dict[str, Any],
    resource_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Build a machine-readable publication-results overview."""

    device_final_validation = device_non_iid["rounds"][-1][
        "validation"
    ]
    device_first_validation = device_non_iid["rounds"][0][
        "validation"
    ]

    iid_final_validation = iid["rounds"][-1]["validation"]
    iid_first_validation = iid["rounds"][0]["validation"]

    return {
        "publication_package": {
            "status": "completed",
            "training_performed": False,
            "evaluation_performed": False,
            "purpose": (
                "Publication-ready packaging of previously "
                "verified experiment results."
            ),
        },
        "centralized": {
            "best_validation_epoch": centralized["best_epoch"],
            "best_validation_f1": centralized[
                "best_validation_f1"
            ],
            "total_training_elapsed_seconds": centralized[
                "total_elapsed_seconds"
            ],
            "final_test": final_test["centralized"],
        },
        "device_non_iid_fedavg": {
            "rounds": device_non_iid["rounds_completed"],
            "round_1_validation_f1": device_first_validation["f1"],
            "round_3_validation_f1": device_final_validation["f1"],
            "round_1_to_round_3_f1_change": (
                device_final_validation["f1"]
                - device_first_validation["f1"]
            ),
            "round_1_validation_roc_auc": device_first_validation[
                "roc_auc"
            ],
            "round_3_validation_roc_auc": device_final_validation[
                "roc_auc"
            ],
            "round_1_to_round_3_roc_auc_change": (
                device_final_validation["roc_auc"]
                - device_first_validation["roc_auc"]
            ),
            "final_test": final_test[
                "device_non_iid_fedavg"
            ],
        },
        "iid_fedavg": {
            "rounds": iid["rounds_completed"],
            "round_1_validation_f1": iid_first_validation["f1"],
            "round_3_validation_f1": iid_final_validation["f1"],
            "round_1_to_round_3_f1_change": (
                iid_final_validation["f1"]
                - iid_first_validation["f1"]
            ),
            "round_1_validation_roc_auc": iid_first_validation[
                "roc_auc"
            ],
            "round_3_validation_roc_auc": iid_final_validation[
                "roc_auc"
            ],
            "round_1_to_round_3_roc_auc_change": (
                iid_final_validation["roc_auc"]
                - iid_first_validation["roc_auc"]
            ),
            "final_test": final_test["iid_fedavg"],
        },
        "local_only": {
            "client_count": final_test["local_only"][
                "client_count"
            ],
            "macro_average": final_test["local_only"][
                "macro_average"
            ],
            "sample_weighted_average": final_test[
                "local_only"
            ]["sample_weighted_average"],
        },
        "communication": {
            "client_count": communication["client_count"],
            "model_parameter_count": communication[
                "model_parameter_count"
            ],
            "model_state_dict_payload_bytes": communication[
                "model_state_dict_payload_bytes"
            ],
            "measured_3_round_total_bytes": communication[
                "measured_3_round_experiment"
            ]["total_bytes"],
            "measured_3_round_total_mb": communication[
                "measured_3_round_experiment"
            ]["total_mb"],
            "configured_10_round_total_bytes": communication[
                "configured_10_round_experiment"
            ]["total_bytes"],
            "configured_10_round_total_mb": communication[
                "configured_10_round_experiment"
            ]["total_mb"],
            "actual_network_transfer": communication[
                "measurement_method"
            ]["actual_network_transfer"],
            "network_traffic_measured": communication[
                "measurement_method"
            ]["network_traffic_measured"],
        },
        "resource_measurements": resource_rows,
        "final_test_summary": {
            "global_test_rows": final_test[
                "frozen_global_test_rows"
            ],
            "local_only_test_rows": final_test[
                "local_only_test_rows_sum"
            ],
            "centralized_f1": final_test["centralized"]["f1"],
            "device_non_iid_fedavg_f1": final_test[
                "device_non_iid_fedavg"
            ]["f1"],
            "iid_fedavg_f1": final_test["iid_fedavg"]["f1"],
            "local_only_macro_f1": final_test["local_only"][
                "macro_average"
            ]["f1"],
            "local_only_weighted_f1": final_test["local_only"][
                "sample_weighted_average"
            ]["f1"],
        },
    }


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------


def main() -> None:
    """Build the complete publication results package."""

    print("=" * 78)
    print("BUILD PUBLICATION RESULTS")
    print("=" * 78)
    print("No model training or evaluation is performed.")
    print()

    print("1. Loading verified result files")

    centralized = load_json(CENTRALIZED_RESULT)
    local_only = load_json(LOCAL_ONLY_RESULT)
    device_non_iid = load_json(DEVICE_NON_IID_RESULT)
    iid = load_json(IID_RESULT)
    communication = load_json(COMMUNICATION_RESULT)
    final_test = load_json(FINAL_TEST_RESULT)

    print("Input result files: PASS")
    print()

    print("2. Validating source results")

    validate_source_results(
        centralized=centralized,
        local_only=local_only,
        device_non_iid=device_non_iid,
        iid=iid,
        communication=communication,
        final_test=final_test,
    )

    print("Source validation: PASS")
    print()

    print("3. Building publication tables")

    TABLES_ROOT.mkdir(parents=True, exist_ok=True)
    FIGURES_ROOT.mkdir(parents=True, exist_ok=True)

    main_performance = build_main_performance(final_test)

    fedavg_convergence = build_fedavg_convergence(
        device_non_iid,
        iid,
    )

    local_only_by_device = build_local_only_by_device(
        final_test
    )

    communication_cost = build_communication_cost(
        communication
    )

    resource_usage = load_resource_metrics()

    validation_to_test = load_validation_to_test_table()

    write_csv(
        main_performance,
        TABLES_ROOT / "main_performance.csv",
    )

    write_csv(
        fedavg_convergence,
        TABLES_ROOT / "fedavg_convergence.csv",
    )

    write_csv(
        local_only_by_device,
        TABLES_ROOT / "local_only_by_device.csv",
    )

    write_csv(
        communication_cost,
        TABLES_ROOT / "communication_cost.csv",
    )

    write_csv(
        resource_usage,
        TABLES_ROOT / "resource_usage.csv",
    )

    write_csv(
        validation_to_test,
        TABLES_ROOT / "validation_to_test.csv",
    )

    print("Publication tables: PASS")
    print()

    print("4. Building publication figures")

    build_fedavg_convergence_figures(
        fedavg_convergence
    )

    build_local_only_f1_figure(
        local_only_by_device
    )

    build_runtime_figure(
        resource_usage
    )

    build_memory_figure(
        resource_usage
    )

    build_communication_figure(
        communication_cost
    )

    print("Publication figures: PASS")
    print()

    print("5. Building publication results JSON")

    publication_analysis = build_publication_analysis(
        centralized=centralized,
        local_only=local_only,
        device_non_iid=device_non_iid,
        iid=iid,
        communication=communication,
        final_test=final_test,
        resource_rows=resource_usage,
    )

    write_json(
        publication_analysis,
        OUTPUT_ROOT / "publication_results.json",
    )

    print("Publication JSON: PASS")
    print()

    print("=" * 78)
    print("PUBLICATION RESULTS BUILD: COMPLETE")
    print("=" * 78)
    print(f"Output directory: {OUTPUT_ROOT}")
    print()
    print(f"Main performance rows: {len(main_performance)}")
    print(f"FedAvg convergence rows: {len(fedavg_convergence)}")
    print(
        f"Local-only device rows: "
        f"{len(local_only_by_device)}"
    )
    print(
        f"Communication rows: "
        f"{len(communication_cost)}"
    )
    print(
        f"Resource rows: "
        f"{len(resource_usage)}"
    )
    print(
        f"Validation-to-test rows: "
        f"{len(validation_to_test)}"
    )
    print()
    print(
        "Centralized final-test F1: "
        f"{final_test['centralized']['f1']:.6f}"
    )
    print(
        "Device Non-IID final-test F1: "
        f"{final_test['device_non_iid_fedavg']['f1']:.6f}"
    )
    print(
        "IID FedAvg final-test F1: "
        f"{final_test['iid_fedavg']['f1']:.6f}"
    )
    print(
        "Local-only macro final-test F1: "
        f"{final_test['local_only']['macro_average']['f1']:.6f}"
    )
    print()
    print(
        "3-round communication payload: "
        f"{communication['measured_3_round_experiment']['total_mb']:.6f} MB"
    )
    print(
        "10-round projected communication payload: "
        f"{communication['configured_10_round_experiment']['total_mb']:.6f} MB"
    )
    print()
    print("All publication artifacts written successfully.")


if __name__ == "__main__":
    main()