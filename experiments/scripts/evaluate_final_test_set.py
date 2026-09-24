from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import joblib
import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.data.torch_data import iter_torch_batches
from src.models.mlp import SmallMLP
from src.models.trainer import EvaluationMetrics, evaluate_batches


DATA_ROOT = Path(r"D:\federated-iot-temp")

SPLIT_ROOT = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
)

SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "training_standard_scaler.pkl"
)

CENTRALIZED_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "centralized"
    / "models"
    / "best_centralized_model.pt"
)

LOCAL_ONLY_MODEL_ROOT = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "local_only"
    / "models"
)

DEVICE_NON_IID_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "device_non_iid_3round_seed42_models"
    / "round_003.pt"
)

IID_MODEL_PATH = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "iid_3round_seed42_models"
    / "round_003.pt"
)

OUTPUT_ROOT = (
    PROJECT_ROOT
    / "results"
    / "raw"
    / "final_test_evaluation"
)

RESULT_JSON_PATH = (
    OUTPUT_ROOT
    / "final_test_evaluation.json"
)

RESULT_CSV_PATH = (
    OUTPUT_ROOT
    / "final_test_evaluation.csv"
)


EXPECTED_TEST_ROWS = 1_059_388
EXPECTED_FEATURE_COUNT = 115
EXPECTED_PARAMETER_COUNT = 9_537

BATCH_SIZE = 256
DEVICE = "cpu"
SEED = 42

CLIENT_IDS = [
    "Danmini_Doorbell",
    "Ecobee_Thermostat",
    "Ennio_Doorbell",
    "Philips_B120N10_Baby_Monitor",
    "Provision_PT_737E_Security_Camera",
    "Provision_PT_838_Security_Camera",
    "Samsung_SNH_1011_N_Webcam",
    "SimpleHome_XCS7_1002_WHT_Security_Camera",
    "SimpleHome_XCS7_1003_WHT_Security_Camera",
]


def load_training_scaler() -> FittedStandardScaler:
    if not SCALER_PATH.is_file():
        raise FileNotFoundError(
            f"Training scaler was not found:\n{SCALER_PATH}"
        )

    scaler = joblib.load(SCALER_PATH)

    if not isinstance(scaler, FittedStandardScaler):
        raise TypeError(
            "Unexpected scaler type: "
            f"{type(scaler).__name__}. "
            "Expected FittedStandardScaler."
        )

    return scaler


def build_model() -> SmallMLP:
    model = SmallMLP(
        input_dim=EXPECTED_FEATURE_COUNT,
        hidden_dim_1=64,
        hidden_dim_2=32,
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in model.parameters()
        if parameter.requires_grad
    )

    if parameter_count != EXPECTED_PARAMETER_COUNT:
        raise RuntimeError(
            "Unexpected model parameter count.\n"
            f"Expected: {EXPECTED_PARAMETER_COUNT:,}\n"
            f"Actual:   {parameter_count:,}"
        )

    return model


def load_checkpoint_state_dict(
    model_path: Path,
    checkpoint_type: str,
) -> dict[str, torch.Tensor]:
    if not model_path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint was not found:\n{model_path}"
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    if checkpoint_type == "checkpoint_dict":
        if not isinstance(checkpoint, dict):
            raise TypeError(
                f"Expected checkpoint dictionary:\n{model_path}"
            )

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "'model_state_dict' was not found in:\n"
                f"{model_path}"
            )

        state_dict = checkpoint["model_state_dict"]

    elif checkpoint_type == "state_dict":
        state_dict = checkpoint

    else:
        raise ValueError(
            f"Unsupported checkpoint type: {checkpoint_type}"
        )

    if not isinstance(state_dict, dict):
        raise TypeError(
            f"Loaded state_dict is not a dictionary:\n{model_path}"
        )

    return state_dict


def load_model(
    model_path: Path,
    checkpoint_type: str,
) -> SmallMLP:
    model = build_model()

    state_dict = load_checkpoint_state_dict(
        model_path=model_path,
        checkpoint_type=checkpoint_type,
    )

    model.load_state_dict(
        state_dict,
        strict=True,
    )

    model.to(DEVICE)
    model.eval()

    return model


def make_test_batches(
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
    devices: list[str] | None = None,
):
    batches = iter_torch_batches(
        loader=loader,
        scaler=scaler,
        split="test",
        devices=devices,
        batch_size=BATCH_SIZE,
        device=DEVICE,
    )

    for batch in batches:
        yield batch.features, batch.labels


def evaluate_model(
    model: SmallMLP,
    loader: NBaIoTSplitLoader,
    scaler: FittedStandardScaler,
    devices: list[str] | None = None,
) -> tuple[EvaluationMetrics, float]:
    criterion = nn.BCELoss()

    start = time.perf_counter()

    metrics = evaluate_batches(
        model=model,
        batches=make_test_batches(
            loader=loader,
            scaler=scaler,
            devices=devices,
        ),
        criterion=criterion,
        device=DEVICE,
    )

    elapsed = time.perf_counter() - start

    return metrics, elapsed


def metrics_to_dict(
    metrics: EvaluationMetrics,
    evaluation_time_seconds: float,
) -> dict:
    return {
        "samples": int(metrics.samples),
        "loss": float(metrics.loss),
        "accuracy": float(metrics.accuracy),
        "precision": float(metrics.precision),
        "recall": float(metrics.recall),
        "f1": float(metrics.f1),
        "roc_auc": float(metrics.roc_auc),
        "evaluation_time_seconds": float(
            evaluation_time_seconds
        ),
    }


def calculate_macro_average(
    client_results: list[dict],
    metric_name: str,
) -> float:
    if not client_results:
        raise ValueError(
            "Cannot calculate macro average with no results."
        )

    return sum(
        float(result[metric_name])
        for result in client_results
    ) / len(client_results)


def calculate_sample_weighted_average(
    client_results: list[dict],
    metric_name: str,
) -> float:
    total_samples = sum(
        int(result["samples"])
        for result in client_results
    )

    if total_samples <= 0:
        raise ValueError(
            "Cannot calculate weighted average with zero samples."
        )

    weighted_sum = sum(
        float(result[metric_name])
        * int(result["samples"])
        for result in client_results
    )

    return weighted_sum / total_samples


def validate_global_test_count(
    loader: NBaIoTSplitLoader,
) -> int:
    test_rows = loader.total_rows_for_split("test")

    if test_rows != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Unexpected global test row count.\n"
            f"Expected: {EXPECTED_TEST_ROWS:,}\n"
            f"Actual:   {test_rows:,}"
        )

    return test_rows


def write_csv(rows: list[dict]) -> None:
    fieldnames = [
        "experiment",
        "evaluation_scope",
        "client_id",
        "samples",
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "evaluation_time_seconds",
    ]

    with RESULT_CSV_PATH.open(
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


def main() -> None:
    torch.set_num_threads(1)
    torch.manual_seed(SEED)

    OUTPUT_ROOT.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("=" * 78)
    print("FINAL HOLDOUT TEST EVALUATION")
    print("=" * 78)
    print(
        f"Project root:       {PROJECT_ROOT}"
    )
    print(
        f"Dataset root:       {DATA_ROOT}"
    )
    print(
        f"Split root:         {SPLIT_ROOT}"
    )
    print(
        f"Scaler:             {SCALER_PATH}"
    )
    print(
        f"Batch size:         {BATCH_SIZE}"
    )
    print(
        f"Expected test rows: {EXPECTED_TEST_ROWS:,}"
    )
    print()

    print("1. Loading N-BaIoT loader")

    loader = NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        data_root=DATA_ROOT,
        split_root=SPLIT_ROOT,
        chunk_size=50_000,
    )

    loader.validate_source_indexes()

    print("Source split indexes: PASS")

    print()
    print("2. Validating frozen test split")

    global_test_rows = validate_global_test_count(loader)

    print(
        f"Global test rows: {global_test_rows:,}"
    )

    print("Frozen global test split: PASS")

    print()
    print("3. Loading training-only scaler")

    scaler = load_training_scaler()

    print(
        f"Scaler type: {type(scaler).__name__}"
    )
    print("Scaler load: PASS")

    print()
    print("4. Verifying model checkpoints")

    required_checkpoints = {
        "centralized": CENTRALIZED_MODEL_PATH,
        "device_non_iid_fedavg": DEVICE_NON_IID_MODEL_PATH,
        "iid_fedavg": IID_MODEL_PATH,
    }

    for experiment, path in required_checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(
                f"{experiment} checkpoint was not found:\n"
                f"{path}"
            )

        print(f"{experiment}: PASS")

    local_only_paths: dict[str, Path] = {}

    for device_id in CLIENT_IDS:
        path = (
            LOCAL_ONLY_MODEL_ROOT
            / f"{device_id}.pt"
        )

        if not path.is_file():
            raise FileNotFoundError(
                f"Local-only checkpoint was not found:\n"
                f"{path}"
            )

        local_only_paths[device_id] = path

    print(
        f"local_only: {len(local_only_paths)}/"
        f"{len(CLIENT_IDS)} checkpoints PASS"
    )

    print()
    print("=" * 78)
    print("5. CENTRALIZED MODEL — GLOBAL TEST SET")
    print("=" * 78)

    centralized_model = load_model(
        model_path=CENTRALIZED_MODEL_PATH,
        checkpoint_type="checkpoint_dict",
    )

    centralized_metrics, centralized_time = evaluate_model(
        model=centralized_model,
        loader=loader,
        scaler=scaler,
        devices=None,
    )

    if centralized_metrics.samples != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Centralized model evaluated on an unexpected "
            "number of test samples.\n"
            f"Expected: {EXPECTED_TEST_ROWS:,}\n"
            f"Actual:   {centralized_metrics.samples:,}"
        )

    centralized_result = metrics_to_dict(
        centralized_metrics,
        centralized_time,
    )

    print(
        f"Samples:   {centralized_metrics.samples:,}"
    )
    print(
        f"Loss:      {centralized_metrics.loss:.12f}"
    )
    print(
        f"Accuracy:  {centralized_metrics.accuracy:.12f}"
    )
    print(
        f"Precision: {centralized_metrics.precision:.12f}"
    )
    print(
        f"Recall:    {centralized_metrics.recall:.12f}"
    )
    print(
        f"F1:        {centralized_metrics.f1:.12f}"
    )
    print(
        f"ROC-AUC:   {centralized_metrics.roc_auc:.12f}"
    )
    print(
        f"Evaluation time: {centralized_time:.2f} seconds"
    )

    print()
    print("=" * 78)
    print("6. DEVICE NON-IID FEDAVG — GLOBAL TEST SET")
    print("=" * 78)

    device_non_iid_model = load_model(
        model_path=DEVICE_NON_IID_MODEL_PATH,
        checkpoint_type="state_dict",
    )

    device_non_iid_metrics, device_non_iid_time = evaluate_model(
        model=device_non_iid_model,
        loader=loader,
        scaler=scaler,
        devices=None,
    )

    if device_non_iid_metrics.samples != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "Device Non-IID FedAvg model evaluated on an "
            "unexpected number of test samples.\n"
            f"Expected: {EXPECTED_TEST_ROWS:,}\n"
            f"Actual:   {device_non_iid_metrics.samples:,}"
        )

    device_non_iid_result = metrics_to_dict(
        device_non_iid_metrics,
        device_non_iid_time,
    )

    print(
        f"Samples:   {device_non_iid_metrics.samples:,}"
    )
    print(
        f"Loss:      {device_non_iid_metrics.loss:.12f}"
    )
    print(
        f"Accuracy:  {device_non_iid_metrics.accuracy:.12f}"
    )
    print(
        f"Precision: {device_non_iid_metrics.precision:.12f}"
    )
    print(
        f"Recall:    {device_non_iid_metrics.recall:.12f}"
    )
    print(
        f"F1:        {device_non_iid_metrics.f1:.12f}"
    )
    print(
        f"ROC-AUC:   {device_non_iid_metrics.roc_auc:.12f}"
    )
    print(
        f"Evaluation time: {device_non_iid_time:.2f} seconds"
    )

    print()
    print("=" * 78)
    print("7. IID FEDAVG — GLOBAL TEST SET")
    print("=" * 78)

    iid_model = load_model(
        model_path=IID_MODEL_PATH,
        checkpoint_type="state_dict",
    )

    iid_metrics, iid_time = evaluate_model(
        model=iid_model,
        loader=loader,
        scaler=scaler,
        devices=None,
    )

    if iid_metrics.samples != EXPECTED_TEST_ROWS:
        raise RuntimeError(
            "IID FedAvg model evaluated on an unexpected "
            "number of test samples.\n"
            f"Expected: {EXPECTED_TEST_ROWS:,}\n"
            f"Actual:   {iid_metrics.samples:,}"
        )

    iid_result = metrics_to_dict(
        iid_metrics,
        iid_time,
    )

    print(
        f"Samples:   {iid_metrics.samples:,}"
    )
    print(
        f"Loss:      {iid_metrics.loss:.12f}"
    )
    print(
        f"Accuracy:  {iid_metrics.accuracy:.12f}"
    )
    print(
        f"Precision: {iid_metrics.precision:.12f}"
    )
    print(
        f"Recall:    {iid_metrics.recall:.12f}"
    )
    print(
        f"F1:        {iid_metrics.f1:.12f}"
    )
    print(
        f"ROC-AUC:   {iid_metrics.roc_auc:.12f}"
    )
    print(
        f"Evaluation time: {iid_time:.2f} seconds"
    )

    print()
    print("=" * 78)
    print("8. LOCAL-ONLY MODELS — SAME-DEVICE TEST SETS")
    print("=" * 78)

    local_only_results: list[dict] = []

    local_only_start = time.perf_counter()

    for client_number, device_id in enumerate(
        CLIENT_IDS,
        start=1,
    ):
        print()
        print(
            f"Device {client_number}/{len(CLIENT_IDS)}: "
            f"{device_id}"
        )

        local_model = load_model(
            model_path=local_only_paths[device_id],
            checkpoint_type="checkpoint_dict",
        )

        local_metrics, local_time = evaluate_model(
            model=local_model,
            loader=loader,
            scaler=scaler,
            devices=[device_id],
        )

        if local_metrics.samples <= 0:
            raise RuntimeError(
                f"No test samples were evaluated for {device_id}."
            )

        result = metrics_to_dict(
            local_metrics,
            local_time,
        )

        result["client_id"] = device_id

        local_only_results.append(result)

        print(
            f"Samples:   {local_metrics.samples:,}"
        )
        print(
            f"Loss:      {local_metrics.loss:.12f}"
        )
        print(
            f"Accuracy:  {local_metrics.accuracy:.12f}"
        )
        print(
            f"Precision: {local_metrics.precision:.12f}"
        )
        print(
            f"Recall:    {local_metrics.recall:.12f}"
        )
        print(
            f"F1:        {local_metrics.f1:.12f}"
        )
        print(
            f"ROC-AUC:   {local_metrics.roc_auc:.12f}"
        )
        print(
            f"Evaluation time: {local_time:.2f} seconds"
        )

    local_only_total_time = (
        time.perf_counter()
        - local_only_start
    )

    local_only_metrics = [
        "loss",
        "accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
    ]

    local_only_macro = {
        metric: calculate_macro_average(
            local_only_results,
            metric,
        )
        for metric in local_only_metrics
    }

    local_only_sample_weighted = {
        metric: calculate_sample_weighted_average(
            local_only_results,
            metric,
        )
        for metric in local_only_metrics
    }

    print()
    print("Local-only macro average:")

    for metric, value in local_only_macro.items():
        print(
            f"  {metric}: {value:.12f}"
        )

    print()
    print("Local-only sample-weighted average:")

    for metric, value in local_only_sample_weighted.items():
        print(
            f"  {metric}: {value:.12f}"
        )

    print()
    print("=" * 78)
    print("9. WRITING RESULTS")
    print("=" * 78)

    csv_rows: list[dict] = []

    global_results = [
        (
            "centralized",
            centralized_result,
        ),
        (
            "device_non_iid_fedavg",
            device_non_iid_result,
        ),
        (
            "iid_fedavg",
            iid_result,
        ),
    ]

    for experiment, result in global_results:
        csv_rows.append(
            {
                "experiment": experiment,
                "evaluation_scope": "global_test",
                "client_id": "",
                "samples": result["samples"],
                "loss": result["loss"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "roc_auc": result["roc_auc"],
                "evaluation_time_seconds": (
                    result["evaluation_time_seconds"]
                ),
            }
        )

    for result in local_only_results:
        csv_rows.append(
            {
                "experiment": "local_only",
                "evaluation_scope": "same_device_test",
                "client_id": result["client_id"],
                "samples": result["samples"],
                "loss": result["loss"],
                "accuracy": result["accuracy"],
                "precision": result["precision"],
                "recall": result["recall"],
                "f1": result["f1"],
                "roc_auc": result["roc_auc"],
                "evaluation_time_seconds": (
                    result["evaluation_time_seconds"]
                ),
            }
        )

    write_csv(csv_rows)

    total_local_test_rows = sum(
        int(result["samples"])
        for result in local_only_results
    )

    total_evaluation_time = (
        centralized_time
        + device_non_iid_time
        + iid_time
        + local_only_total_time
    )

    result = {
        "experiment": "final_test_evaluation",
        "status": "completed",
        "dataset": "N-BaIoT",
        "model": "small_mlp",
        "seed": SEED,
        "device": DEVICE,
        "batch_size": BATCH_SIZE,
        "model_parameter_count": EXPECTED_PARAMETER_COUNT,
        "frozen_global_test_rows": EXPECTED_TEST_ROWS,
        "local_only_test_rows_sum": total_local_test_rows,
        "scaler": {
            "path": str(SCALER_PATH),
            "type": type(scaler).__name__,
            "fitted_on_training_data_only": True,
            "refit_on_test_data": False,
        },
        "centralized": {
            "checkpoint_path": str(
                CENTRALIZED_MODEL_PATH
            ),
            "evaluation_scope": "global_test",
            **centralized_result,
        },
        "device_non_iid_fedavg": {
            "checkpoint_path": str(
                DEVICE_NON_IID_MODEL_PATH
            ),
            "evaluation_scope": "global_test",
            "round": 3,
            **device_non_iid_result,
        },
        "iid_fedavg": {
            "checkpoint_path": str(
                IID_MODEL_PATH
            ),
            "evaluation_scope": "global_test",
            "round": 3,
            **iid_result,
        },
        "local_only": {
            "checkpoint_root": str(
                LOCAL_ONLY_MODEL_ROOT
            ),
            "evaluation_scope": "same_device_test",
            "client_count": len(CLIENT_IDS),
            "clients": local_only_results,
            "macro_average": local_only_macro,
            "sample_weighted_average": (
                local_only_sample_weighted
            ),
            "total_evaluation_time_seconds": (
                local_only_total_time
            ),
        },
        "outputs": {
            "json": str(RESULT_JSON_PATH),
            "csv": str(RESULT_CSV_PATH),
        },
        "total_evaluation_time_seconds": (
            total_evaluation_time
        ),
        "methodological_notes": [
            (
                "This is a final holdout evaluation. "
                "No model training or parameter updates "
                "are performed by this script."
            ),
            (
                "The test split is the frozen test partition "
                "created before model evaluation."
            ),
            (
                "The centralized model and both FedAvg models "
                "are evaluated on the global test set."
            ),
            (
                "Each local-only model is evaluated only on "
                "the test rows belonging to its corresponding "
                "N-BaIoT device."
            ),
            (
                "The persisted standard scaler was fitted "
                "using centralized training data and reused "
                "during test evaluation."
            ),
            (
                "The scaler was not refitted on the test set."
            ),
            (
                "Because the preprocessing scaler was fitted "
                "centrally, the federated experiments do not "
                "represent a fully decentralized preprocessing "
                "pipeline."
            ),
            (
                "N-BaIoT devices represent simulated logical "
                "IoT clients rather than physical deployments."
            ),
            (
                "Federated learning does not by itself provide "
                "differential privacy or secure aggregation."
            ),
            (
                "Communication measurements reported elsewhere "
                "are model-tensor payload calculations rather "
                "than measurements of actual network traffic."
            ),
            (
                "Local-only results use same-device test "
                "evaluation, whereas centralized and FedAvg "
                "results use the global test set."
            ),
            (
                "Only seed 42 was evaluated; cross-seed "
                "variability has not been measured."
            ),
        ],
    }

    RESULT_JSON_PATH.write_text(
        json.dumps(
            result,
            indent=2,
        ),
        encoding="utf-8",
    )

    if not RESULT_JSON_PATH.is_file():
        raise RuntimeError(
            "JSON output was not created."
        )

    if not RESULT_CSV_PATH.is_file():
        raise RuntimeError(
            "CSV output was not created."
        )

    print()
    print("=" * 78)
    print("FINAL HOLDOUT TEST EVALUATION: COMPLETE")
    print("=" * 78)
    print()
    print(
        f"Global test rows verified: "
        f"{EXPECTED_TEST_ROWS:,}"
    )
    print(
        f"Local-only test rows evaluated: "
        f"{total_local_test_rows:,}"
    )
    print()
    print(
        f"Centralized F1: "
        f"{centralized_result['f1']:.6f}"
    )
    print(
        f"Device Non-IID F1: "
        f"{device_non_iid_result['f1']:.6f}"
    )
    print(
        f"IID FedAvg F1: "
        f"{iid_result['f1']:.6f}"
    )
    print()
    print(
        f"Local-only macro F1: "
        f"{local_only_macro['f1']:.6f}"
    )
    print(
        f"Local-only weighted F1: "
        f"{local_only_sample_weighted['f1']:.6f}"
    )
    print()
    print(
        f"JSON: {RESULT_JSON_PATH}"
    )
    print(
        f"CSV:  {RESULT_CSV_PATH}"
    )
    print(
        f"Total evaluation time: "
        f"{total_evaluation_time:.2f} seconds"
    )


if __name__ == "__main__":
    main()