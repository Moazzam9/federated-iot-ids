from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.nbaiot_loader import NBaIoTSplitLoader


EXPECTED_TRAIN_ROWS = 4_943_824

DEVICES = [
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

SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "splits"
    / "source_index_summary.csv"
)

DATA_ROOT = PROJECT_ROOT.parent / "federated-iot-temp"


def load_expected_device_counts() -> dict[str, int]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Source-index summary was not found:\n"
            f"{SUMMARY_PATH}"
        )

    counts = {
        device: 0
        for device in DEVICES
    }

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        rows = csv.DictReader(handle)

        for row in rows:
            source_file = row["source_file"]
            device = Path(source_file).parts[0]

            if device not in counts:
                raise AssertionError(
                    f"Unexpected device in source summary: {device}"
                )

            counts[device] += int(row["train_rows"])

    total = sum(counts.values())

    if total != EXPECTED_TRAIN_ROWS:
        raise AssertionError(
            "Expected device training-row counts to sum to "
            f"{EXPECTED_TRAIN_ROWS:,}, found {total:,}."
        )

    return counts


def main() -> None:
    print("REAL N-BAIoT DEVICE LOADER FILTER VERIFICATION")
    print()

    print(f"Dataset root: {DATA_ROOT}")
    print(f"Source summary: {SUMMARY_PATH}")
    print()

    expected_counts = load_expected_device_counts()

    print("EXPECTED TRAINING ROW COUNTS")
    print()

    for device in DEVICES:
        print(
            f"{device}: "
            f"{expected_counts[device]:,}"
        )

    print()
    print("CREATING N-BAIoT LOADER")
    print()

    loader = NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        data_root=DATA_ROOT,
        chunk_size=50_000,
    )

    loader.validate_source_indexes()

    print("Source split indexes: PASS")
    print()

    actual_total = 0

    print("VERIFYING EACH DEVICE")
    print()

    for device in DEVICES:
        expected = expected_counts[device]

        print(f"Device: {device}")
        print(
            f"Expected training rows: {expected:,}"
        )

        actual_rows = 0
        chunk_count = 0

        for chunk in loader.iter_chunks(
            split="train",
            devices=[device],
        ):
            chunk_count += 1
            actual_rows += len(chunk.features)

            unique_devices = set(
                chunk.devices.tolist()
            )

            if unique_devices != {device}:
                raise AssertionError(
                    "Device filtering failure.\n"
                    f"Requested device: {device}\n"
                    f"Returned devices: {sorted(unique_devices)}"
                )

            if chunk.split != "train":
                raise AssertionError(
                    f"Unexpected split returned: {chunk.split}"
                )

            if len(chunk.features) != len(
                chunk.binary_labels
            ):
                raise AssertionError(
                    f"Feature/label length mismatch for {device}."
                )

            if len(chunk.features) != len(
                chunk.devices
            ):
                raise AssertionError(
                    f"Feature/device length mismatch for {device}."
                )

        print(
            f"Actual training rows:   {actual_rows:,}"
        )
        print(f"Chunks read:             {chunk_count:,}")

        if actual_rows != expected:
            raise AssertionError(
                f"Row-count mismatch for {device}.\n"
                f"Expected: {expected:,}\n"
                f"Actual:   {actual_rows:,}"
            )

        print("Device filtering:        PASS")
        print("Row-count verification:  PASS")
        print()

        actual_total += actual_rows

    print("GLOBAL VERIFICATION")
    print()

    print(
        f"Expected total training rows: {EXPECTED_TRAIN_ROWS:,}"
    )
    print(
        f"Actual total training rows:   {actual_total:,}"
    )

    if actual_total != EXPECTED_TRAIN_ROWS:
        raise AssertionError(
            "Global device-filtered training rows do not "
            "match the expected training-row count."
        )

    print("Global row-count verification: PASS")
    print()
    print("DEVICE LOADER FILTER VERIFICATION: PASS")
    print()
    print(
        "Each logical device client retrieves only its own "
        "N-BaIoT training rows."
    )
    print(
        "The loader-returned row counts match the independent "
        "source-index counts for all nine devices."
    )


if __name__ == "__main__":
    main()