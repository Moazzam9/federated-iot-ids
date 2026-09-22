from __future__ import annotations

import json
import sys
from dataclasses import fields
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import torch


# ----------------------------------------------------------------------
# Make project root importable.
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ----------------------------------------------------------------------
# Project imports.
# ----------------------------------------------------------------------

from src.data.preprocessing import FittedStandardScaler
from src.fl.client import ClientTrainingResult
from src.models.mlp import SmallMLP
from src.models.trainer import EvaluationMetrics


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

SCRIPT_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "scripts"
    / "run_multiround_federated.py"
)

EXPECTED_CLIENT_IDS = [
    "client_1",
    "client_2",
    "client_3",
    "client_4",
    "client_5",
    "client_6",
    "client_7",
    "client_8",
    "client_9",
]

EXPECTED_CLIENT_COUNTS = {
    "client_1": 112,
    "client_2": 111,
    "client_3": 111,
    "client_4": 111,
    "client_5": 111,
    "client_6": 111,
    "client_7": 111,
    "client_8": 111,
    "client_9": 111,
}

EXPECTED_TOTAL_SAMPLES = 1000
EXPECTED_VALIDATION_SAMPLES = 200

EXPECTED_LOSS = 0.10
EXPECTED_ACCURACY = 0.95
EXPECTED_PRECISION = 0.94
EXPECTED_RECALL = 0.93
EXPECTED_F1 = 0.935
EXPECTED_ROC_AUC = 0.97

EXPECTED_CLIENT_LOSS = 0.25
EXPECTED_CLIENT_EPOCHS = 1
EXPECTED_CLIENT_ELAPSED = 0.01
EXPECTED_AGGREGATION_TIME = 0.01


# ----------------------------------------------------------------------
# Fake loader
# ----------------------------------------------------------------------

class FakeLoader:
    """
    Minimal fake N-BaIoT loader.

    The production runner only needs the attributes and method defined
    here during the smoke test.
    """

    def __init__(self) -> None:
        self.training_row_count = EXPECTED_TOTAL_SAMPLES
        self.validation_row_count = EXPECTED_VALIDATION_SAMPLES

        self.source_records = []

    def validate_source_indexes(self) -> None:
        return None


def fake_create_loader(*args, **kwargs) -> FakeLoader:
    return FakeLoader()


# ----------------------------------------------------------------------
# Fake scaler
# ----------------------------------------------------------------------

def make_fake_scaler() -> FittedStandardScaler:
    """
    Construct a minimal FittedStandardScaler instance.

    No real dataset or real scaler file is loaded.
    """

    scaler = object.__new__(FittedStandardScaler)

    scaler.scaler = None
    scaler.feature_count = 115
    scaler.fitted = True

    return scaler


def fake_load_training_scaler(
    *args,
    **kwargs,
) -> FittedStandardScaler:
    return make_fake_scaler()


# ----------------------------------------------------------------------
# Fake client data source
# ----------------------------------------------------------------------

class FakeClientDataSource:
    """
    Deterministic fake client data source.

    112 + (8 * 111) = 1000 total samples.
    """

    def __init__(self) -> None:
        self.client_counts = dict(EXPECTED_CLIENT_COUNTS)

    def count_training_samples(
        self,
        client_id: str,
    ) -> int:
        if client_id not in self.client_counts:
            raise ValueError(
                f"Unknown fake client: {client_id}"
            )

        return self.client_counts[client_id]


def fake_create_client_data_source(
    *args,
    **kwargs,
):
    """
    Exact return structure used by the production runner:

        client_ids, client_data_source
    """

    return (
        EXPECTED_CLIENT_IDS.copy(),
        FakeClientDataSource(),
    )


# ----------------------------------------------------------------------
# State-dict helper
# ----------------------------------------------------------------------

def clone_state_dict(
    state_dict: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    return {
        key: value.detach().cpu().clone()
        for key, value in state_dict.items()
    }


# ----------------------------------------------------------------------
# Fake ClientTrainingResult
# ----------------------------------------------------------------------

def make_client_training_result(
    client_id: str,
    sample_count: int,
    state_dict: dict[str, torch.Tensor],
) -> ClientTrainingResult:
    """
    Build ClientTrainingResult using its actual dataclass fields.

    The production runner explicitly accesses:

        client_id
        samples
        epochs
        loss
        elapsed_seconds

    The state dictionary is also supplied where the actual dataclass
    exposes a state/model-state field.
    """

    result_values = {}

    for field in fields(ClientTrainingResult):
        name = field.name

        if name == "client_id":
            result_values[name] = client_id

        elif name == "samples":
            result_values[name] = sample_count

        elif name == "epochs":
            result_values[name] = EXPECTED_CLIENT_EPOCHS

        elif name == "loss":
            result_values[name] = EXPECTED_CLIENT_LOSS

        elif name == "elapsed_seconds":
            result_values[name] = EXPECTED_CLIENT_ELAPSED

        elif name == "state_dict":
            result_values[name] = clone_state_dict(
                state_dict
            )

        elif name == "model_state_dict":
            result_values[name] = clone_state_dict(
                state_dict
            )

    return ClientTrainingResult(
        **result_values
    )


# ----------------------------------------------------------------------
# Fake FedAvg round
# ----------------------------------------------------------------------

def fake_run_federated_round(
    *args,
    **kwargs,
):
    """
    Deterministic replacement for run_federated_round().

    This allows the real multi-round orchestration to execute without
    training 4.9 million real rows.
    """

    global_model = kwargs.get(
        "global_model"
    )

    if global_model is None and args:
        global_model = args[0]

    if global_model is None:
        global_model = SmallMLP()

    round_number = kwargs.get(
        "round_number",
        1,
    )

    global_state = clone_state_dict(
        global_model.state_dict()
    )

    client_results = []

    for client_id in EXPECTED_CLIENT_IDS:
        client_result = make_client_training_result(
            client_id=client_id,
            sample_count=EXPECTED_CLIENT_COUNTS[
                client_id
            ],
            state_dict=global_state,
        )

        client_results.append(
            client_result
        )

    from src.fl.coordinator import (
        FederatedRoundResult,
    )

    result_values = {}

    for field in fields(FederatedRoundResult):
        name = field.name

        if name == "round_number":
            result_values[name] = round_number

        elif name == "client_results":
            result_values[name] = client_results

        elif name == "global_state_dict":
            result_values[name] = global_state

        elif name == "total_client_samples":
            result_values[name] = (
                EXPECTED_TOTAL_SAMPLES
            )

        elif name == "aggregation_elapsed_seconds":
            result_values[name] = (
                EXPECTED_AGGREGATION_TIME
            )

    return FederatedRoundResult(
        **result_values
    )


# ----------------------------------------------------------------------
# Fake validation
# ----------------------------------------------------------------------

def fake_evaluate_global_model(
    *args,
    **kwargs,
):
    """
    Exact return structure expected by the production runner:

        validation_metrics, validation_time
    """

    metrics = EvaluationMetrics(
        loss=EXPECTED_LOSS,
        samples=EXPECTED_VALIDATION_SAMPLES,
        accuracy=EXPECTED_ACCURACY,
        precision=EXPECTED_PRECISION,
        recall=EXPECTED_RECALL,
        f1=EXPECTED_F1,
        roc_auc=EXPECTED_ROC_AUC,
    )

    return metrics, 0.01


# ----------------------------------------------------------------------
# Numeric helper
# ----------------------------------------------------------------------

def assert_close(
    actual,
    expected,
    name: str,
    tolerance: float = 1e-12,
) -> None:
    if abs(
        float(actual) - float(expected)
    ) > tolerance:
        raise AssertionError(
            f"{name}: expected "
            f"{expected}, got {actual}"
        )


# ----------------------------------------------------------------------
# Validate one round
# ----------------------------------------------------------------------

def validate_round(
    round_data: dict,
    expected_round_number: int,
) -> None:

    assert (
        round_data["round_number"]
        == expected_round_number
    )

    assert (
        round_data["client_count"]
        == len(EXPECTED_CLIENT_IDS)
    )

    assert (
        round_data["total_client_samples"]
        == EXPECTED_TOTAL_SAMPLES
    )

    assert_close(
        round_data[
            "aggregation_elapsed_seconds"
        ],
        EXPECTED_AGGREGATION_TIME,
        "aggregation_elapsed_seconds",
    )

    assert (
        "measured_wall_time_seconds"
        in round_data
    )

    assert (
        round_data[
            "measured_wall_time_seconds"
        ]
        >= 0
    )

    assert (
        "checkpoint_path"
        in round_data
    )

    # --------------------------------------------------------------
    # Validation metrics
    # --------------------------------------------------------------

    validation = round_data[
        "validation"
    ]

    assert (
        validation["samples"]
        == EXPECTED_VALIDATION_SAMPLES
    )

    assert_close(
        validation["loss"],
        EXPECTED_LOSS,
        "validation.loss",
    )

    assert_close(
        validation["accuracy"],
        EXPECTED_ACCURACY,
        "validation.accuracy",
    )

    assert_close(
        validation["precision"],
        EXPECTED_PRECISION,
        "validation.precision",
    )

    assert_close(
        validation["recall"],
        EXPECTED_RECALL,
        "validation.recall",
    )

    assert_close(
        validation["f1"],
        EXPECTED_F1,
        "validation.f1",
    )

    assert_close(
        validation["roc_auc"],
        EXPECTED_ROC_AUC,
        "validation.roc_auc",
    )

    assert (
        validation[
            "evaluation_time_seconds"
        ]
        == 0.01
    )

    # --------------------------------------------------------------
    # Per-client records
    # --------------------------------------------------------------

    clients = round_data[
        "clients"
    ]

    assert isinstance(
        clients,
        list,
    )

    assert len(clients) == 9

    actual_client_ids = [
        client["client_id"]
        for client in clients
    ]

    assert (
        actual_client_ids
        == EXPECTED_CLIENT_IDS
    )

    for client in clients:
        client_id = client[
            "client_id"
        ]

        assert (
            client["samples"]
            == EXPECTED_CLIENT_COUNTS[
                client_id
            ]
        )

        assert (
            client["epochs"]
            == EXPECTED_CLIENT_EPOCHS
        )

        assert_close(
            client["loss"],
            EXPECTED_CLIENT_LOSS,
            f"{client_id}.loss",
        )

        assert_close(
            client["elapsed_seconds"],
            EXPECTED_CLIENT_ELAPSED,
            f"{client_id}.elapsed_seconds",
        )


# ----------------------------------------------------------------------
# Main smoke test
# ----------------------------------------------------------------------

def test_runner_smoke() -> None:

    print("=" * 72)
    print("MULTI-ROUND FEDAVG SMOKE TEST")
    print("=" * 72)

    if not SCRIPT_PATH.is_file():
        raise FileNotFoundError(
            "Production runner does not exist:\n"
            f"{SCRIPT_PATH}"
        )

    # Import the actual production runner.
    import experiments.scripts.run_multiround_federated as runner

    with TemporaryDirectory(
        prefix="multiround_smoke_"
    ) as temp_dir:

        temp_root = Path(temp_dir)

        # IMPORTANT:
        #
        # The production runner itself creates:
        #
        #     MODEL_ROOT / f"{output_prefix}_models"
        #
        # Therefore MODEL_ROOT must point to temp_root here.
        #
        # If MODEL_ROOT were already set to
        # temp_root / "smoke_test_multiround_models",
        # the production runner would incorrectly produce:
        #
        #     smoke_test_multiround_models/
        #         smoke_test_multiround_models/
        #
        model_root = temp_root

        # ----------------------------------------------------------
        # Patch only the expensive/data-dependent operations.
        # The actual runner.main() remains under test.
        # ----------------------------------------------------------

        patches = [
            patch.object(
                runner,
                "create_loader",
                fake_create_loader,
            ),
            patch.object(
                runner,
                "load_training_scaler",
                fake_load_training_scaler,
            ),
            patch.object(
                runner,
                "create_client_data_source",
                fake_create_client_data_source,
            ),
            patch.object(
                runner,
                "run_federated_round",
                fake_run_federated_round,
            ),
            patch.object(
                runner,
                "evaluate_global_model",
                fake_evaluate_global_model,
            ),
        ]

        # Redirect model output.
        #
        # The production runner will create:
        #
        #     temp_root / "smoke_test_multiround_models"
        #
        patches.append(
            patch.object(
                runner,
                "MODEL_ROOT",
                model_root,
            )
        )

        # Redirect JSON output.
        patches.append(
            patch.object(
                runner,
                "RESULTS_ROOT",
                temp_root,
            )
        )

        for current_patch in patches:
            current_patch.start()

        original_argv = sys.argv.copy()

        try:
            sys.argv = [
                str(SCRIPT_PATH),
                "--partition",
                "iid",
                "--rounds",
                "2",
                "--local-epochs",
                "1",
                "--batch-size",
                "256",
                "--learning-rate",
                "0.001",
                "--seed",
                "42",
                "--output-prefix",
                "smoke_test_multiround",
            ]

            runner.main()

        finally:
            sys.argv = original_argv

            for current_patch in reversed(
                patches
            ):
                current_patch.stop()

        # ----------------------------------------------------------
        # Find result JSON.
        # ----------------------------------------------------------

        result_candidates = list(
            temp_root.rglob(
                "smoke_test_multiround.json"
            )
        )

        if not result_candidates:
            raise AssertionError(
                "Smoke-test result JSON was not created."
            )

        result_path = result_candidates[0]

        print()
        print(
            f"Smoke result JSON: "
            f"{result_path}"
        )

        # ----------------------------------------------------------
        # Load result.
        # ----------------------------------------------------------

        with result_path.open(
            "r",
            encoding="utf-8",
        ) as handle:
            result = json.load(handle)

        # ----------------------------------------------------------
        # Validate exact top-level fields written by runner.
        # ----------------------------------------------------------

        assert (
            result["experiment"]
            == "smoke_test_multiround"
        )

        assert (
            result["status"]
            == "completed"
        )

        assert (
            result["partition_type"]
            == "iid"
        )

        assert (
            result["seed"]
            == 42
        )

        assert (
            result["clients"]
            == 9
        )

        assert (
            result["client_ids"]
            == EXPECTED_CLIENT_IDS
        )

        assert (
            result["local_epochs"]
            == 1
        )

        assert (
            result["batch_size"]
            == 256
        )

        assert_close(
            result["learning_rate"],
            0.001,
            "learning_rate",
        )

        assert (
            result["device"]
            == "cpu"
        )

        assert (
            result["model_parameter_count"]
            == 9537
        )

        assert (
            result["validation_max_batches"]
            is None
        )

        assert (
            result["rounds_requested"]
            == 2
        )

        assert (
            result["rounds_completed"]
            == 2
        )

        # This is the ACTUAL field name in the runner.
        assert (
            "total_experiment_wall_time_seconds"
            in result
        )

        assert (
            result[
                "total_experiment_wall_time_seconds"
            ]
            >= 0
        )

        assert (
            "rounds"
            in result
        )

        assert isinstance(
            result["rounds"],
            list,
        )

        assert (
            len(result["rounds"])
            == 2
        )

        assert (
            "notes"
            in result
        )

        assert isinstance(
            result["notes"],
            list,
        )

        # ----------------------------------------------------------
        # Validate round 1.
        # ----------------------------------------------------------

        validate_round(
            result["rounds"][0],
            expected_round_number=1,
        )

        # ----------------------------------------------------------
        # Validate round 2.
        # ----------------------------------------------------------

        validate_round(
            result["rounds"][1],
            expected_round_number=2,
        )

        # ----------------------------------------------------------
        # Validate checkpoint files.
        #
        # The production runner creates:
        #
        #     MODEL_ROOT / f"{output_prefix}_models"
        #
        # Since MODEL_ROOT == temp_root, the actual directory is:
        #
        #     temp_root/
        #         smoke_test_multiround_models/
        #
        # ----------------------------------------------------------

        checkpoint_directory = (
            temp_root
            / "smoke_test_multiround_models"
        )

        checkpoint_1 = (
            checkpoint_directory
            / "round_001.pt"
        )

        checkpoint_2 = (
            checkpoint_directory
            / "round_002.pt"
        )

        assert checkpoint_1.is_file(), (
            "Round 1 checkpoint missing:\n"
            f"{checkpoint_1}"
        )

        assert checkpoint_2.is_file(), (
            "Round 2 checkpoint missing:\n"
            f"{checkpoint_2}"
        )

        # ----------------------------------------------------------
        # Validate actual checkpoint format.
        #
        # The production runner saves:
        #
        #     model.state_dict()
        #
        # so each checkpoint should be a state dictionary that can
        # be loaded into SmallMLP.
        # ----------------------------------------------------------

        checkpoint_model_1 = SmallMLP()

        state_dict_1 = torch.load(
            checkpoint_1,
            map_location="cpu",
            weights_only=True,
        )

        checkpoint_model_1.load_state_dict(
            state_dict_1,
            strict=True,
        )

        checkpoint_model_2 = SmallMLP()

        state_dict_2 = torch.load(
            checkpoint_2,
            map_location="cpu",
            weights_only=True,
        )

        checkpoint_model_2.load_state_dict(
            state_dict_2,
            strict=True,
        )

        # ----------------------------------------------------------
        # Final success.
        # ----------------------------------------------------------

        print()
        print("=" * 72)
        print("MULTI-ROUND FEDAVG SMOKE TEST: PASS")
        print("=" * 72)
        print("Rounds tested:          2")
        print("Clients per round:      9")
        print("Total fake samples:     1,000")
        print("Validation samples:     200")
        print("Round 1 checkpoint:     PASS")
        print("Round 2 checkpoint:     PASS")
        print("Checkpoint reload:      PASS")
        print("Client result records:  PASS")
        print("Result JSON validation: PASS")
        print("=" * 72)


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    test_runner_smoke()