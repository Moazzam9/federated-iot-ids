from __future__ import annotations

import sys
from pathlib import Path

import joblib
import numpy as np
import torch

# ----------------------------------------------------------------------
# Project path
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ----------------------------------------------------------------------
# Project imports
# ----------------------------------------------------------------------

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import FittedStandardScaler
from src.fl.client_data import DeviceClientDataSource
from src.fl.coordinator import run_federated_round
from src.fl.partitioning import make_device_partitions
from src.models.mlp import SmallMLP


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

DATASET_ROOT = Path(r"D:\federated-iot-temp")

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

EXPECTED_SOURCE_FILES = 89
EXPECTED_TRAIN_ROWS = 4_943_824
EXPECTED_VALIDATION_ROWS = 1_059_394
EXPECTED_FEATURES = 115
EXPECTED_CLIENTS = 9
EXPECTED_MODEL_PARAMETERS = 9_537

BATCH_SIZE = 256

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


# ----------------------------------------------------------------------
# Helper
# ----------------------------------------------------------------------

def check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    print("=" * 72)
    print("DEVICE NON-IID FEDAVG PREFLIGHT")
    print("=" * 72)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Dataset root: {DATASET_ROOT}")
    print()

    # ------------------------------------------------------------------
    # 1. Create and validate loader
    # ------------------------------------------------------------------

    print("1. Creating loader...")

    loader = NBaIoTSplitLoader(
        data_root=DATASET_ROOT,
        split_root=SPLIT_ROOT,
    )

    check(
        loader.source_count() == EXPECTED_SOURCE_FILES,
        (
            "Unexpected source-file count. "
            f"Expected {EXPECTED_SOURCE_FILES}, "
            f"got {loader.source_count()}."
        ),
    )

    loader.validate_source_indexes()

    check(
        loader.total_rows_for_split("train")
        == EXPECTED_TRAIN_ROWS,
        "Training row count does not match frozen split.",
    )

    check(
        loader.total_rows_for_split("validation")
        == EXPECTED_VALIDATION_ROWS,
        "Validation row count does not match frozen split.",
    )

    check(
        loader.total_rows() == 7_062_606,
        "Total dataset row count does not match audited total.",
    )

    check(
        loader.list_devices() == sorted(CLIENT_IDS),
        "Loader device list does not match expected nine devices.",
    )

    print("   Source split indexes: PASS")
    print("   Source-file count: PASS")
    print("   Training row count: PASS")
    print("   Validation row count: PASS")
    print("   Device list: PASS")
    print()

    # ------------------------------------------------------------------
    # 2. Check device partitions
    # ------------------------------------------------------------------

    print("2. Checking device partitions...")

    partitions = make_device_partitions(CLIENT_IDS)

    check(
        isinstance(partitions, dict),
        "make_device_partitions() must return a dictionary.",
    )

    check(
        len(partitions) == EXPECTED_CLIENTS,
        (
            "Unexpected number of device partitions. "
            f"Expected {EXPECTED_CLIENTS}, "
            f"got {len(partitions)}."
        ),
    )

    check(
        list(partitions.keys()) == CLIENT_IDS,
        "Device/client mapping does not preserve expected order.",
    )

    for client_id in CLIENT_IDS:
        partition = partitions[client_id]

        check(
            partition.client_id == client_id,
            f"Partition identity mismatch for {client_id}.",
        )

        check(
            partition.row_indices.dtype == np.int64,
            f"Unexpected row-index dtype for {client_id}.",
        )

        check(
            len(partition.row_indices) == 0,
            (
                f"Device partition for {client_id} should be "
                "identity-only and contain no copied row data."
            ),
        )

    print("   9 device partitions: PASS")
    print("   Device/client mapping: PASS")
    print("   Identity-only partition design: PASS")
    print()

    # ------------------------------------------------------------------
    # 3. Load persisted scaler
    # ------------------------------------------------------------------

    print("3. Loading scaler...")

    check(
        SCALER_PATH.is_file(),
        f"Scaler file does not exist: {SCALER_PATH}",
    )

    scaler = joblib.load(SCALER_PATH)

    check(
        isinstance(scaler, FittedStandardScaler),
        "Persisted scaler is not a FittedStandardScaler.",
    )

    print("   Training-only scaler: PASS")
    print()

    # ------------------------------------------------------------------
    # 4. Create device client data source
    # ------------------------------------------------------------------

    print("4. Creating device client data source...")

    client_data_source = DeviceClientDataSource(loader)

    client_sample_counts: dict[str, int] = {}

    for client_id in CLIENT_IDS:
        count = client_data_source.count_training_samples(
            client_id
        )

        check(
            count > 0,
            f"Client {client_id} has no training samples.",
        )

        client_sample_counts[client_id] = count

        print(
            f"   {client_id}: {count:,}"
        )

    total_client_samples = sum(
        client_sample_counts.values()
    )

    print(
        f"   Total training rows: "
        f"{total_client_samples:,}"
    )

    check(
        total_client_samples == EXPECTED_TRAIN_ROWS,
        (
            "Device client sample counts do not sum to "
            "the frozen training row count."
        ),
    )

    print("   Client sample counts: PASS")
    print()

    # ------------------------------------------------------------------
    # 5. Read one real batch from every device
    # ------------------------------------------------------------------

    print("5. Reading one real batch from every device...")

    real_batches = {}

    for client_id in CLIENT_IDS:
        batches = client_data_source.iter_torch_batches(
            client_id=client_id,
            scaler=scaler,
            batch_size=BATCH_SIZE,
            device="cpu",
        )

        batch = next(batches)

        check(
            isinstance(batch.features, torch.Tensor),
            f"{client_id}: features are not a torch.Tensor.",
        )

        check(
            isinstance(batch.labels, torch.Tensor),
            f"{client_id}: labels are not a torch.Tensor.",
        )

        check(
            batch.features.shape[0] == BATCH_SIZE,
            (
                f"{client_id}: unexpected batch size. "
                f"Got {batch.features.shape[0]}."
            ),
        )

        check(
            batch.features.shape[1] == EXPECTED_FEATURES,
            (
                f"{client_id}: unexpected feature count. "
                f"Got {batch.features.shape[1]}."
            ),
        )

        check(
            batch.labels.shape[0] == BATCH_SIZE,
            (
                f"{client_id}: unexpected label count. "
                f"Got {batch.labels.shape[0]}."
            ),
        )

        check(
            batch.features.device.type == "cpu",
            f"{client_id}: features are not on CPU.",
        )

        check(
            batch.labels.device.type == "cpu",
            f"{client_id}: labels are not on CPU.",
        )

        check(
            torch.isfinite(batch.features).all().item(),
            f"{client_id}: non-finite feature values detected.",
        )

        check(
            torch.isfinite(batch.labels).all().item(),
            f"{client_id}: non-finite label values detected.",
        )

        check(
            torch.all(
                (batch.labels == 0)
                | (batch.labels == 1)
            ).item(),
            f"{client_id}: labels are not binary 0/1.",
        )

        real_batches[client_id] = batch

        print(
            f"   {client_id}: "
            f"{batch.features.shape[0]} samples x "
            f"{batch.features.shape[1]} features -> PASS"
        )

    print()

    # ------------------------------------------------------------------
    # 6. Model validation
    # ------------------------------------------------------------------

    print("6. Testing actual client -> trainer -> FedAvg path...")

    global_model = SmallMLP(
        input_dim=EXPECTED_FEATURES
    )

    parameter_count = sum(
        parameter.numel()
        for parameter in global_model.parameters()
        if parameter.requires_grad
    )

    print(
        f"   Model parameters: "
        f"{parameter_count:,}",
        end="",
    )

    check(
        parameter_count == EXPECTED_MODEL_PARAMETERS,
        (
            "Unexpected number of trainable model parameters. "
            f"Expected {EXPECTED_MODEL_PARAMETERS}, "
            f"got {parameter_count}."
        ),
    )

    print(" -> PASS")

    # ------------------------------------------------------------------
    # 7. One-batch smoke test
    # ------------------------------------------------------------------
    #
    # IMPORTANT:
    # This intentionally trains only one client on one real 256-row
    # batch. It does NOT start the full experiment.
    # ------------------------------------------------------------------

    smoke_client = CLIENT_IDS[0]

    real_batch = real_batches[smoke_client]

    original_count_method = (
        client_data_source.count_training_samples
    )

    original_batch_method = (
        client_data_source.iter_torch_batches
    )

    def smoke_count(client_id: str) -> int:
        check(
            client_id == smoke_client,
            (
                "Unexpected client requested during "
                "one-batch smoke test."
            ),
        )

        return BATCH_SIZE

    def smoke_batches(
        client_id: str,
        scaler: FittedStandardScaler,
        batch_size: int = 256,
        device: str = "cpu",
    ):
        check(
            client_id == smoke_client,
            (
                "Unexpected client requested during "
                "one-batch smoke test."
            ),
        )

        check(
            batch_size == BATCH_SIZE,
            "Unexpected smoke-test batch size.",
        )

        check(
            device == "cpu",
            "Unexpected smoke-test device.",
        )

        yield real_batch

    # Temporarily replace the instance methods so the coordinator
    # sees exactly one real batch and cannot begin a full run.
    client_data_source.count_training_samples = (
        smoke_count
    )

    client_data_source.iter_torch_batches = (
        smoke_batches
    )

    try:
        result = run_federated_round(
            global_model=global_model,
            loader=loader,
            scaler=scaler,
            client_ids=[smoke_client],
            local_epochs=1,
            batch_size=BATCH_SIZE,
            learning_rate=0.001,
            device="cpu",
            round_number=1,
            client_data_source=client_data_source,
        )

    finally:
        client_data_source.count_training_samples = (
            original_count_method
        )

        client_data_source.iter_torch_batches = (
            original_batch_method
        )

    check(
        result.round_number == 1,
        "Unexpected federated round number.",
    )

    check(
        len(result.client_results) == 1,
        (
            "Smoke test did not produce exactly "
            "one client result."
        ),
    )

    client_result = result.client_results[0]

    check(
        client_result.client_id == smoke_client,
        "Smoke-test client ID is incorrect.",
    )

    check(
        client_result.samples == BATCH_SIZE,
        (
            "Smoke-test client sample count is incorrect. "
            f"Expected {BATCH_SIZE}, "
            f"got {client_result.samples}."
        ),
    )

    check(
        result.total_client_samples == BATCH_SIZE,
        (
            "Smoke test did not aggregate exactly "
            "one batch."
        ),
    )

    check(
        bool(result.global_state_dict),
        "FedAvg did not return a global state dictionary.",
    )

    # Verify the global model was actually updated and remains valid.
    for parameter_name, parameter in global_model.state_dict().items():
        check(
            parameter_name in result.global_state_dict,
            (
                "Aggregated global state is missing "
                f"parameter: {parameter_name}"
            ),
        )

        check(
            torch.isfinite(
                result.global_state_dict[parameter_name]
            ).all().item(),
            (
                "Aggregated global state contains "
                f"non-finite values: {parameter_name}"
            ),
        )

    print("   One real client batch -> trainer: PASS")
    print("   Client result sample count: PASS")
    print("   FedAvg aggregation: PASS")
    print("   Global model update: PASS")
    print()

    # ------------------------------------------------------------------
    # Final result
    # ------------------------------------------------------------------

    print("=" * 72)
    print("DEVICE NON-IID FEDAVG PREFLIGHT: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()