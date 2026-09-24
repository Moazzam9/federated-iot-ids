from __future__ import annotations

from pathlib import Path
import sys

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "configs" / "experiment.yaml"


EXPECTED_DEVICES = 9
EXPECTED_FEATURES = 115
EXPECTED_SOURCE_FILES = 89
EXPECTED_TOTAL_ROWS = 7_062_606

VALID_SPLITS = {
    "train": 0,
    "validation": 1,
    "test": 2,
}

VALID_BASELINES = {
    "centralized",
    "local_only",
    "federated",
}

VALID_FEDERATED_ALGORITHMS = {
    "fedavg",
}

VALID_ATTACK_FAMILIES = {
    "benign",
    "gafgyt",
    "mirai",
}


def require(condition: bool, message: str) -> None:
    """Raise a clear error when a configuration requirement fails."""
    if not condition:
        raise ValueError(message)


def main() -> None:
    print("=" * 70)
    print("EXPERIMENT CONFIGURATION VALIDATION")
    print("=" * 70)
    print()

    print(f"Project root:  {PROJECT_ROOT}")
    print(f"Config file:   {CONFIG_PATH}")
    print()

    require(
        CONFIG_PATH.exists(),
        f"Configuration file does not exist:\n{CONFIG_PATH}",
    )

    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)

    require(
        isinstance(config, dict),
        "Configuration root must be a YAML mapping.",
    )

    # ------------------------------------------------------------------
    # Project
    # ------------------------------------------------------------------

    project = config.get("project", {})

    require(
        project.get("name") == "federated-iot-ids",
        "Project name must be 'federated-iot-ids'.",
    )

    require(
        project.get("random_seed") == 42,
        "Random seed must be 42.",
    )

    print("Project configuration: PASS")

    # ------------------------------------------------------------------
    # Dataset
    # ------------------------------------------------------------------

    dataset = config.get("dataset", {})

    require(
        dataset.get("name") == "N-BaIoT",
        "Dataset must be N-BaIoT.",
    )

    require(
        dataset.get("expected_source_files") == EXPECTED_SOURCE_FILES,
        f"Expected source-file count must be {EXPECTED_SOURCE_FILES}.",
    )

    require(
        dataset.get("expected_total_rows") == EXPECTED_TOTAL_ROWS,
        f"Expected total rows must be {EXPECTED_TOTAL_ROWS:,}.",
    )

    require(
        dataset.get("expected_features") == EXPECTED_FEATURES,
        f"Expected feature count must be {EXPECTED_FEATURES}.",
    )

    require(
        dataset.get("expected_devices") == EXPECTED_DEVICES,
        f"Expected device count must be {EXPECTED_DEVICES}.",
    )

    splits = dataset.get("splits", {})

    require(
        splits == VALID_SPLITS,
        f"Dataset split codes must be exactly {VALID_SPLITS}.",
    )

    labels = dataset.get("labels", {})

    require(
        labels.get("benign") == 0,
        "Benign label must be 0.",
    )

    require(
        labels.get("attack") == 1,
        "Attack label must be 1.",
    )

    attack_families = set(dataset.get("attack_families", []))

    require(
        attack_families == VALID_ATTACK_FAMILIES,
        (
            "Attack families must be exactly "
            f"{sorted(VALID_ATTACK_FAMILIES)}."
        ),
    )

    client_definition = dataset.get("client_definition", {})

    require(
        client_definition.get("type") == "device",
        "FL clients must be defined by device.",
    )

    print("Dataset configuration: PASS")

    # ------------------------------------------------------------------
    # Preprocessing
    # ------------------------------------------------------------------

    preprocessing = config.get("preprocessing", {})

    require(
        preprocessing.get("method") == "standard_scaler",
        "Preprocessing method must be StandardScaler.",
    )

    require(
        preprocessing.get("scaler_fit_split") == "train_only",
        "Scaler must be fitted using training data only.",
    )

    require(
        preprocessing.get("feature_count") == EXPECTED_FEATURES,
        f"Preprocessing feature count must be {EXPECTED_FEATURES}.",
    )

    require(
        preprocessing.get("missing_value_policy") == "fail",
        "Missing-value policy must be 'fail'.",
    )

    print("Preprocessing configuration: PASS")

    # ------------------------------------------------------------------
    # Model
    # ------------------------------------------------------------------

    model = config.get("model", {})

    require(
        model.get("name") == "small_mlp",
        "Primary model must be small_mlp.",
    )

    require(
        model.get("framework") == "pytorch",
        "Model framework must be PyTorch.",
    )

    architecture = model.get("architecture", {})

    require(
        architecture.get("input_dim") == EXPECTED_FEATURES,
        f"Model input dimension must be {EXPECTED_FEATURES}.",
    )

    require(
        architecture.get("hidden_layers") == [64, 32],
        "Hidden layers must be [64, 32].",
    )

    require(
        architecture.get("output_dim") == 1,
        "Model output dimension must be 1.",
    )

    require(
        architecture.get("activation") == "relu",
        "Hidden activation must be ReLU.",
    )

    require(
        architecture.get("output_activation") == "sigmoid",
        "Output activation must be sigmoid.",
    )

    require(
        model.get("loss") == "binary_cross_entropy",
        "Loss must be binary cross entropy.",
    )

    require(
        model.get("optimizer") == "adam",
        "Optimizer must be Adam.",
    )

    print("Model configuration: PASS")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    training = config.get("training", {})

    require(
        training.get("batch_size", 0) > 0,
        "Batch size must be greater than zero.",
    )

    require(
        training.get("local_epochs", 0) > 0,
        "Local epochs must be greater than zero.",
    )

    require(
        training.get("max_centralized_epochs", 0) > 0,
        "Centralized epoch limit must be greater than zero.",
    )

    require(
        training.get("max_local_only_epochs", 0) > 0,
        "Local-only epoch limit must be greater than zero.",
    )

    require(
        training.get("max_federated_rounds", 0) > 0,
        "Federated round limit must be greater than zero.",
    )

    print("Training configuration: PASS")

    # ------------------------------------------------------------------
    # Federated learning
    # ------------------------------------------------------------------

    federated = config.get("federated", {})

    require(
        federated.get("algorithm") in VALID_FEDERATED_ALGORITHMS,
        "Federated algorithm must be FedAvg.",
    )

    require(
        federated.get("client_count") == EXPECTED_DEVICES,
        f"Federated client count must be {EXPECTED_DEVICES}.",
    )

    require(
        federated.get("clients_per_round") == EXPECTED_DEVICES,
        f"All {EXPECTED_DEVICES} clients must participate per round.",
    )

    federated_settings = set(federated.get("settings", []))

    require(
        federated_settings == {"iid", "device_non_iid"},
        "Federated settings must contain IID and device_non_iid.",
    )

    aggregation = federated.get("aggregation", {})

    require(
        aggregation.get("weighting") == "number_of_training_samples",
        "FedAvg weighting must use training-sample counts.",
    )

    print("Federated-learning configuration: PASS")

    # ------------------------------------------------------------------
    # Baselines
    # ------------------------------------------------------------------

    baselines = set(config.get("baselines", []))

    require(
        baselines == VALID_BASELINES,
        (
            "Baselines must be exactly "
            f"{sorted(VALID_BASELINES)}."
        ),
    )

    print("Baseline configuration: PASS")

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    evaluation = config.get("evaluation", {})

    primary_metrics = set(
        evaluation.get("primary_metrics", [])
    )

    required_primary_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    }

    require(
        primary_metrics == required_primary_metrics,
        (
            "Primary metrics must be exactly "
            f"{sorted(required_primary_metrics)}."
        ),
    )

    additional_metrics = set(
        evaluation.get("additional_metrics", [])
    )

    require(
        "confusion_matrix" in additional_metrics,
        "Confusion matrix must be included.",
    )

    validation_usage = evaluation.get("validation_usage", {})

    require(
        validation_usage.get("purpose")
        == "model_selection_and_configuration",
        "Validation set must be reserved for model selection/configuration.",
    )

    require(
        validation_usage.get("test_usage")
        == "final_evaluation_only",
        "Test set must be reserved for final evaluation.",
    )

    print("Evaluation configuration: PASS")

    # ------------------------------------------------------------------
    # Communication
    # ------------------------------------------------------------------

    communication = config.get("communication", {})
    communication_metrics = set(
        communication.get("measure", [])
    )

    required_communication_metrics = {
        "model_parameter_count",
        "upload_bytes",
        "download_bytes",
        "total_bytes",
    }

    require(
        communication_metrics == required_communication_metrics,
        (
            "Communication metrics must be exactly "
            f"{sorted(required_communication_metrics)}."
        ),
    )

    print("Communication configuration: PASS")

    # ------------------------------------------------------------------
    # Privacy
    # ------------------------------------------------------------------

    privacy = config.get("privacy", {})

    require(
        privacy.get("differential_privacy") is False,
        "Differential privacy must currently be disabled.",
    )

    require(
        privacy.get("secure_aggregation") is False,
        "Secure aggregation must currently be disabled.",
    )

    require(
        privacy.get("privacy_claim")
        == "Federated learning alone does not guarantee privacy.",
        "Privacy claim does not match the frozen methodological wording.",
    )

    print("Privacy configuration: PASS")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------

    reproducibility = config.get("reproducibility", {})

    require(
        reproducibility.get("seed") == 42,
        "Reproducibility seed must be 42.",
    )

    require(
        reproducibility.get("deterministic") is True,
        "Deterministic execution must be enabled.",
    )

    require(
        reproducibility.get("save_config") is True,
        "Configuration saving must be enabled.",
    )

    require(
        reproducibility.get("save_metrics") is True,
        "Metric saving must be enabled.",
    )

    print("Reproducibility configuration: PASS")

    print()
    print("=" * 70)
    print("STAGE 5G.20D.4 CONFIGURATION VALIDATION: PASS")
    print("=" * 70)
    print()
    print("The experiment contract is internally consistent.")
    print("No experiment has been executed by this validator.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print()
        print("=" * 70)
        print("CONFIGURATION VALIDATION: FAIL")
        print("=" * 70)
        print()
        print(f"Error: {exc}")
        sys.exit(1)