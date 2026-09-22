from __future__ import annotations

import json
import tempfile
from dataclasses import fields
from pathlib import Path

import torch

from src.models.mlp import SmallMLP
from src.models.trainer import EvaluationMetrics
from src.fl.client import ClientTrainingResult
from src.fl.coordinator import FederatedRoundResult


# ----------------------------------------------------------------------
# Import production runner
# ----------------------------------------------------------------------

import experiments.scripts.run_multiround_federated as runner


# ----------------------------------------------------------------------
# Fake loader
# ----------------------------------------------------------------------

class FakeLoader:
    """
    Minimal loader interface required by the production runner.

    This fake loader allows the real multi-round runner to be tested
    without reading the full N-BaIoT dataset.
    """

    def __init__(
        self,
        training_rows: int = 1000,
        validation_rows: int = 200,
    ) -> None:

        self._training_rows = training_rows
        self._validation_rows = validation_rows

        self.source_records = []

    def validate_source_indexes(self) -> None:
        return None

    def total_rows_for_split(
        self,
        split: str,
    ) -> int:

        if split == "train":
            return self._training_rows

        if split == "validation":
            return self._validation_rows

        raise ValueError(
            f"Unsupported fake split: {split!r}"
        )


# ----------------------------------------------------------------------
# Fake client data source
# ----------------------------------------------------------------------

class FakeClientDataSource:
    """
    Deterministic nine-client fake data source.

    The sample counts sum to exactly 1,000.
    """

    CLIENT_COUNTS = {
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

    def count_training_samples(
        self,
        client_id: str,
    ) -> int:

        if client_id not in self.CLIENT_COUNTS:
            raise ValueError(
                f"Unknown fake client: {client_id!r}"
            )

        return self.CLIENT_COUNTS[client_id]


# ----------------------------------------------------------------------
# Fake scaler
# ----------------------------------------------------------------------

class FakeScaler:
    pass


# ----------------------------------------------------------------------
# Fake client result
# ----------------------------------------------------------------------

def make_fake_client_result(
    client_id: str,
    samples: int,
    epochs: int,
    global_model: SmallMLP,
) -> ClientTrainingResult:
    """
    Construct a fake ClientTrainingResult using the actual
    ClientTrainingResult dataclass from src.fl.client.

    The smoke test does not perform real client training.
    """

    field_names = {
        field.name
        for field in fields(ClientTrainingResult)
    }

    values = {}

    if "client_id" in field_names:
        values["client_id"] = client_id

    if "samples" in field_names:
        values["samples"] = samples

    if "epochs" in field_names:
        values["epochs"] = epochs

    if "loss" in field_names:
        values["loss"] = 0.25

    if "elapsed_seconds" in field_names:
        values["elapsed_seconds"] = 0.01

    state_dict = {
        key: value.detach().clone()
        for key, value in global_model.state_dict().items()
    }

    if "state_dict" in field_names:
        values["state_dict"] = state_dict

    if "model_state_dict" in field_names:
        values["model_state_dict"] = state_dict

    return ClientTrainingResult(**values)


# ----------------------------------------------------------------------
# Fake FedAvg round
# ----------------------------------------------------------------------

def fake_run_federated_round(
    global_model,
    loader,
    scaler,
    client_ids,
    local_epochs=1,
    batch_size=256,
    learning_rate=0.001,
    device="cpu",
    client_train_fn=None,
    round_number=1,
    client_data_source=None,
):
    """
    Fake one-round FedAvg implementation.

    The real production runner calls this function during the smoke
    test. The function returns the real FederatedRoundResult type.
    """

    client_results = []

    for client_id in client_ids:

        samples = (
            client_data_source.count_training_samples(
                client_id
            )
        )

        client_results.append(
            make_fake_client_result(
                client_id=client_id,
                samples=samples,
                epochs=local_epochs,
                global_model=global_model,
            )
        )

    global_state = {
        key: value.detach().clone()
        for key, value in global_model.state_dict().items()
    }

    return FederatedRoundResult(
        round_number=round_number,
        client_results=client_results,
        global_state_dict=global_state,
        total_client_samples=sum(
            result.samples
            for result in client_results
        ),
        aggregation_elapsed_seconds=0.01,
    )


# ----------------------------------------------------------------------
# Fake validation
# ----------------------------------------------------------------------

def fake_evaluate_global_model(
    model,
    loader,
    scaler,
    batch_size,
    max_batches,
):
    """
    Return deterministic validation metrics.

    These values are smoke-test placeholders only and must never be
    used as experimental results.
    """

    metrics = EvaluationMetrics(
        loss=0.10,
        samples=200,
        accuracy=0.95,
        precision=0.94,
        recall=0.93,
        f1=0.935,
        roc_auc=0.97,
    )

    return metrics, 0.01


# ----------------------------------------------------------------------
# Smoke test
# ----------------------------------------------------------------------

def test_runner_smoke() -> None:

    print("=" * 72)
    print("MULTI-ROUND FEDAVG SMOKE TEST")
    print("=" * 72)

    temp_root = Path(
        tempfile.mkdtemp(
            prefix="multiround_smoke_"
        )
    )

    result_path = (
        temp_root
        / "smoke_test_multiround.json"
    )

    model_root = temp_root

    # --------------------------------------------------------------
    # Save original runner functions/paths
    # --------------------------------------------------------------

    original_create_loader = (
        runner.create_loader
    )

    original_load_training_scaler = (
        runner.load_training_scaler
    )

    original_create_client_data_source = (
        runner.create_client_data_source
    )

    original_run_federated_round = (
        runner.run_federated_round
    )

    original_evaluate_global_model = (
        runner.evaluate_global_model
    )

    original_results_root = (
        runner.RESULTS_ROOT
    )

    original_model_root = (
        runner.MODEL_ROOT
    )

    original_parse_args = (
        runner.parse_args
    )

    try:

        # ----------------------------------------------------------
        # Create fake components
        # ----------------------------------------------------------

        fake_loader = FakeLoader(
            training_rows=1000,
            validation_rows=200,
        )

        fake_scaler = FakeScaler()

        fake_client_data_source = (
            FakeClientDataSource()
        )

        # ----------------------------------------------------------
        # Patch data-dependent components
        # ----------------------------------------------------------

        runner.create_loader = (
            lambda: fake_loader
        )

        runner.load_training_scaler = (
            lambda: fake_scaler
        )

        runner.create_client_data_source = (
            lambda loader, partition: (
                [
                    "client_1",
                    "client_2",
                    "client_3",
                    "client_4",
                    "client_5",
                    "client_6",
                    "client_7",
                    "client_8",
                    "client_9",
                ],
                fake_client_data_source,
            )
        )

        runner.run_federated_round = (
            fake_run_federated_round
        )

        runner.evaluate_global_model = (
            fake_evaluate_global_model
        )

        # ----------------------------------------------------------
        # Patch output paths
        # ----------------------------------------------------------

        runner.RESULTS_ROOT = temp_root
        runner.MODEL_ROOT = model_root

        # ----------------------------------------------------------
        # Patch command-line arguments
        # ----------------------------------------------------------

        class FakeArgs:
            partition = "iid"
            rounds = 2
            local_epochs = 1
            batch_size = 256
            learning_rate = 0.001
            seed = 42
            validation_max_batches = None
            output_prefix = (
                "smoke_test_multiround"
            )

        runner.parse_args = (
            lambda: FakeArgs()
        )

        # ----------------------------------------------------------
        # Execute the REAL production runner
        # ----------------------------------------------------------

        runner.main()

        # ----------------------------------------------------------
        # Validate result JSON
        # ----------------------------------------------------------

        if not result_path.is_file():
            raise AssertionError(
                "Smoke-test result JSON was not created:\n"
                f"{result_path}"
            )

        result = json.loads(
            result_path.read_text(
                encoding="utf-8"
            )
        )

        if result["status"] != "completed":
            raise AssertionError(
                "Expected completed experiment status."
            )

        if result["partition_type"] != "iid":
            raise AssertionError(
                "Unexpected partition type."
            )

        if result["rounds_requested"] != 2:
            raise AssertionError(
                "Expected two requested rounds."
            )

        if result["rounds_completed"] != 2:
            raise AssertionError(
                "Expected two completed rounds."
            )

        if result["clients"] != 9:
            raise AssertionError(
                "Expected nine clients."
            )

        if result["total_training_rows"] != 1000:
            raise AssertionError(
                "Expected 1,000 training rows."
            )

        if result["total_validation_rows"] != 200:
            raise AssertionError(
                "Expected 200 validation rows."
            )

        if result["client_sample_counts"]["client_1"] != 112:
            raise AssertionError(
                "Unexpected client_1 sample count."
            )

        if sum(
            result["client_sample_counts"].values()
        ) != 1000:
            raise AssertionError(
                "Client sample counts do not sum to 1,000."
            )

        # ----------------------------------------------------------
        # Validate rounds
        # ----------------------------------------------------------

        rounds = result["rounds"]

        if len(rounds) != 2:
            raise AssertionError(
                "Expected exactly two round records."
            )

        for expected_round, round_result in enumerate(
            rounds,
            start=1,
        ):

            if round_result["round_number"] != expected_round:
                raise AssertionError(
                    "Unexpected round number."
                )

            if round_result["client_count"] != 9:
                raise AssertionError(
                    "Expected nine clients in every round."
                )

            if round_result[
                "total_client_samples"
            ] != 1000:
                raise AssertionError(
                    "Expected 1,000 samples in every round."
                )

            if len(
                round_result["clients"]
            ) != 9:
                raise AssertionError(
                    "Expected nine client result records."
                )

            validation = (
                round_result["validation"]
            )

            if validation["samples"] != 200:
                raise AssertionError(
                    "Expected 200 validation samples."
                )

            checkpoint_path = Path(
                round_result["checkpoint_path"]
            )

            if not checkpoint_path.is_file():
                raise AssertionError(
                    "Expected checkpoint does not exist:\n"
                    f"{checkpoint_path}"
                )

        # ----------------------------------------------------------
        # Validate checkpoints
        # ----------------------------------------------------------

        model_directory = (
            temp_root
            / "smoke_test_multiround_models"
        )

        checkpoint_1 = (
            model_directory
            / "round_001.pt"
        )

        checkpoint_2 = (
            model_directory
            / "round_002.pt"
        )

        if not checkpoint_1.is_file():
            raise AssertionError(
                "Round 1 checkpoint missing."
            )

        if not checkpoint_2.is_file():
            raise AssertionError(
                "Round 2 checkpoint missing."
            )

        model_1 = SmallMLP()

        state_1 = torch.load(
            checkpoint_1,
            map_location="cpu",
            weights_only=True,
        )

        model_1.load_state_dict(
            state_1,
            strict=True,
        )

        model_2 = SmallMLP()

        state_2 = torch.load(
            checkpoint_2,
            map_location="cpu",
            weights_only=True,
        )

        model_2.load_state_dict(
            state_2,
            strict=True,
        )

        # ----------------------------------------------------------
        # Success
        # ----------------------------------------------------------

        print()
        print(
            f"Smoke result JSON: "
            f"{result_path}"
        )

        print()
        print("=" * 72)
        print(
            "MULTI-ROUND FEDAVG SMOKE TEST: PASS"
        )
        print("=" * 72)

        print(
            "Rounds tested:          2"
        )
        print(
            "Clients per round:      9"
        )
        print(
            "Total fake samples:     1,000"
        )
        print(
            "Validation samples:     200"
        )
        print(
            "Round 1 checkpoint:     PASS"
        )
        print(
            "Round 2 checkpoint:     PASS"
        )
        print(
            "Checkpoint reload:      PASS"
        )
        print(
            "Client result records:  PASS"
        )
        print(
            "Result JSON validation: PASS"
        )
        print("=" * 72)

    finally:

        # ----------------------------------------------------------
        # Restore runner
        # ----------------------------------------------------------

        runner.create_loader = (
            original_create_loader
        )

        runner.load_training_scaler = (
            original_load_training_scaler
        )

        runner.create_client_data_source = (
            original_create_client_data_source
        )

        runner.run_federated_round = (
            original_run_federated_round
        )

        runner.evaluate_global_model = (
            original_evaluate_global_model
        )

        runner.RESULTS_ROOT = (
            original_results_root
        )

        runner.MODEL_ROOT = (
            original_model_root
        )

        runner.parse_args = (
            original_parse_args
        )


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

if __name__ == "__main__":
    test_runner_smoke()