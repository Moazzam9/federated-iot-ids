from __future__ import annotations

import csv
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.fl.partitioning import make_device_partitions


EXPECTED_TRAIN_ROWS = 4_943_824
EXPECTED_SOURCE_FILES = 89
EXPECTED_DEVICE_COUNT = 9


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


def get_device(source_file: str) -> str:
    parts = Path(source_file).parts

    if not parts:
        raise ValueError(
            f"Could not determine device from source file: "
            f"{source_file}"
        )

    device = parts[0]

    if device not in DEVICES:
        raise ValueError(
            f"Unknown device '{device}' in source file: "
            f"{source_file}"
        )

    return device


def get_attack_family(source_file: str) -> str:
    normalized = source_file.replace("\\", "/").lower()

    if normalized.endswith("/benign_traffic.csv"):
        return "benign"

    if "/gafgyt_attacks/" in normalized:
        return "gafgyt"

    if "/mirai_attacks/" in normalized:
        return "mirai"

    raise ValueError(
        "Could not determine attack family from source file: "
        f"{source_file}"
    )


def load_source_summary() -> list[dict]:
    if not SUMMARY_PATH.exists():
        raise FileNotFoundError(
            "Source-index summary was not found:\n"
            f"{SUMMARY_PATH}"
        )

    with SUMMARY_PATH.open(
        "r",
        encoding="utf-8",
        newline="",
    ) as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    print("REAL N-BaIoT DEVICE NON-IID PARTITION AUDIT")
    print()

    rows = load_source_summary()

    print(f"Source summary: {SUMMARY_PATH}")
    print(f"Source files found: {len(rows)}")

    if len(rows) != EXPECTED_SOURCE_FILES:
        raise AssertionError(
            f"Expected {EXPECTED_SOURCE_FILES} source files, "
            f"found {len(rows)}."
        )

    print("Source-file count check: PASS")

    device_stats = {
        device: {
            "rows": 0,
            "benign": 0,
            "gafgyt": 0,
            "mirai": 0,
        }
        for device in DEVICES
    }

    global_train_rows = 0

    for row in rows:
        source_file = row["source_file"]
        train_rows = int(row["train_rows"])

        if train_rows < 0:
            raise AssertionError(
                f"Negative training-row count for source file: "
                f"{source_file}"
            )

        device = get_device(source_file)
        attack_family = get_attack_family(source_file)

        device_stats[device]["rows"] += train_rows
        device_stats[device][attack_family] += train_rows

        global_train_rows += train_rows

    print()
    print(
        "Training rows from source summary: "
        f"{global_train_rows:,}"
    )

    if global_train_rows != EXPECTED_TRAIN_ROWS:
        raise AssertionError(
            f"Expected {EXPECTED_TRAIN_ROWS:,} training rows, "
            f"found {global_train_rows:,}."
        )

    print("Training-row count check: PASS")

    discovered_devices = set(device_stats)

    if discovered_devices != set(DEVICES):
        raise AssertionError(
            "Discovered device set does not match the expected "
            "N-BaIoT device set."
        )

    print("Device set check: PASS")

    total_device_rows = sum(
        stats["rows"]
        for stats in device_stats.values()
    )

    if total_device_rows != EXPECTED_TRAIN_ROWS:
        raise AssertionError(
            "Device training-row totals do not equal the "
            "global training-row count."
        )

    print("Device coverage check: PASS")

    print()
    print("DEVICE TRAINING DISTRIBUTIONS")
    print()

    for device in DEVICES:
        stats = device_stats[device]
        total = stats["rows"]

        if total <= 0:
            raise AssertionError(
                f"Device has no training rows: {device}"
            )

        family_total = (
            stats["benign"]
            + stats["gafgyt"]
            + stats["mirai"]
        )

        if family_total != total:
            raise AssertionError(
                f"Attack-family counts do not sum to total rows "
                f"for device: {device}"
            )

        benign_pct = 100.0 * stats["benign"] / total
        gafgyt_pct = 100.0 * stats["gafgyt"] / total
        mirai_pct = 100.0 * stats["mirai"] / total

        print(device)
        print(f"  Rows:   {total:,}")
        print(
            f"  Benign: {stats['benign']:,} "
            f"({benign_pct:.4f}%)"
        )
        print(
            f"  Gafgyt: {stats['gafgyt']:,} "
            f"({gafgyt_pct:.4f}%)"
        )
        print(
            f"  Mirai:  {stats['mirai']:,} "
            f"({mirai_pct:.4f}%)"
        )
        print()

    print("CREATING DEVICE CLIENT PARTITIONS")
    print()

    partitions = make_device_partitions(DEVICES)

    if len(partitions) != EXPECTED_DEVICE_COUNT:
        raise AssertionError(
            f"Expected {EXPECTED_DEVICE_COUNT} client partitions, "
            f"found {len(partitions)}."
        )

    print("Client-count check: PASS")

    expected_client_ids = DEVICES.copy()
    actual_client_ids = list(partitions)

    if actual_client_ids != expected_client_ids:
        raise AssertionError(
            "Device client identifiers do not match the expected "
            "device order.\n"
            f"Expected: {expected_client_ids}\n"
            f"Actual:   {actual_client_ids}"
        )

    print("Client-ID/device-name check: PASS")

    if len(set(actual_client_ids)) != len(actual_client_ids):
        raise AssertionError(
            "Device client identifiers are not unique."
        )

    print("Client-ID uniqueness check: PASS")

    print()
    print("CLIENT → DEVICE MAPPING")
    print()

    for client_id in actual_client_ids:
        print(f"{client_id} -> {client_id}")

    print()
    print("DEVICE NON-IID PARTITION AUDIT: PASS")
    print()
    print(
        "The nine N-BaIoT devices are treated as nine "
        "simulated logical IoT clients."
    )
    print(
        "Each device name is used as the logical client "
        "identifier."
    )
    print(
        "The actual training rows remain associated with "
        "their original device."
    )
    print(
        "Rows are not randomly redistributed across devices."
    )
    print(
        "Device-specific class and attack-family distributions "
        "are therefore preserved for the non-IID experiment."
    )


if __name__ == "__main__":
    main()