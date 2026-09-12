from pathlib import Path
import sys

# ---------------------------------------------------------------------------
# Make the project root importable when this script is executed directly.
# ---------------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Project import
# ---------------------------------------------------------------------------

from src.data.nbaiot_loader import NBaIoTSplitLoader


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_ROOT = Path(r"D:\federated-iot-temp")


# ---------------------------------------------------------------------------
# Main verification
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("N-BaIoT SPLIT-AWARE DATA LOADER VERIFICATION")
    print("=" * 70)

    print()
    print(f"Project root: {PROJECT_ROOT}")
    print(f"Data root:    {DATA_ROOT}")

    # -----------------------------------------------------------------------
    # Create loader
    # -----------------------------------------------------------------------

    print()
    print("Creating N-BaIoT split loader...")

    loader = NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        data_root=DATA_ROOT,
        chunk_size=10_000,
    )

    print("Loader creation: PASS")

    # -----------------------------------------------------------------------
    # Basic dataset information
    # -----------------------------------------------------------------------

    print()
    print("Dataset information")
    print("-" * 70)

    source_count = loader.source_count()
    total_rows = loader.total_rows()

    train_rows = loader.total_rows_for_split(
        "train"
    )

    validation_rows = loader.total_rows_for_split(
        "validation"
    )

    test_rows = loader.total_rows_for_split(
        "test"
    )

    print(f"Source files:     {source_count:,}")
    print(f"Total rows:       {total_rows:,}")
    print(f"Train rows:       {train_rows:,}")
    print(f"Validation rows:  {validation_rows:,}")
    print(f"Test rows:        {test_rows:,}")

    # -----------------------------------------------------------------------
    # Expected frozen counts
    # -----------------------------------------------------------------------

    expected_sources = 89
    expected_total = 7_062_606
    expected_train = 4_943_824
    expected_validation = 1_059_394
    expected_test = 1_059_388

    print()
    print("Checking frozen dataset counts...")
    print("-" * 70)

    if source_count != expected_sources:
        raise RuntimeError(
            f"Source-file count mismatch: "
            f"expected {expected_sources}, "
            f"found {source_count}"
        )

    if total_rows != expected_total:
        raise RuntimeError(
            f"Total row count mismatch: "
            f"expected {expected_total:,}, "
            f"found {total_rows:,}"
        )

    if train_rows != expected_train:
        raise RuntimeError(
            f"Train row count mismatch: "
            f"expected {expected_train:,}, "
            f"found {train_rows:,}"
        )

    if validation_rows != expected_validation:
        raise RuntimeError(
            f"Validation row count mismatch: "
            f"expected {expected_validation:,}, "
            f"found {validation_rows:,}"
        )

    if test_rows != expected_test:
        raise RuntimeError(
            f"Test row count mismatch: "
            f"expected {expected_test:,}, "
            f"found {test_rows:,}"
        )

    print("Source-file count: PASS")
    print("Frozen split counts: PASS")

    # -----------------------------------------------------------------------
    # Device list
    # -----------------------------------------------------------------------

    print()
    print("Logical FL clients / devices")
    print("-" * 70)

    devices = loader.list_devices()

    for number, device in enumerate(devices, start=1):
        print(f"{number:2d}. {device}")

    if len(devices) != 9:
        raise RuntimeError(
            f"Expected 9 logical devices, found {len(devices)}"
        )

    print()
    print("Device count: PASS")

    # -----------------------------------------------------------------------
    # Validate source indexes
    # -----------------------------------------------------------------------

    print()
    print("Validating source split indexes...")
    print("-" * 70)

    loader.validate_source_indexes()

    print("Source index validation: PASS")

    # -----------------------------------------------------------------------
    # Training sample
    # -----------------------------------------------------------------------

    print()
    print("Reading a training sample...")
    print("-" * 70)

    train_sample = loader.sample(
        split="train",
        n_rows=10,
    )

    print(
        f"Sample feature shape: "
        f"{train_sample.features.shape}"
    )

    print(
        f"Sample labels:        "
        f"{train_sample.binary_labels.tolist()}"
    )

    print(
        f"Sample attack family: "
        f"{train_sample.attack_families.tolist()}"
    )

    print(
        f"Sample source file:    "
        f"{train_sample.source_file}"
    )

    print(
        f"Sample row numbers:    "
        f"{train_sample.row_numbers.tolist()}"
    )

    # -----------------------------------------------------------------------
    # Feature validation
    # -----------------------------------------------------------------------

    print()
    print("Checking feature structure...")
    print("-" * 70)

    feature_count = train_sample.features.shape[1]

    print(f"Feature count: {feature_count}")

    if feature_count != 115:
        raise RuntimeError(
            f"Expected 115 numerical features, "
            f"found {feature_count}"
        )

    print("Feature count: PASS")

    # -----------------------------------------------------------------------
    # Feature names
    # -----------------------------------------------------------------------

    print()
    print("First 10 feature names:")
    print("-" * 70)

    for number, column in enumerate(
        train_sample.features.columns[:10],
        start=1,
    ):
        print(f"{number:2d}. {column}")

    # -----------------------------------------------------------------------
    # Label validation
    # -----------------------------------------------------------------------

    unique_labels = set(
        train_sample.binary_labels.tolist()
    )

    print()
    print(
        "Unique binary labels in sample: "
        f"{sorted(unique_labels)}"
    )

    if not unique_labels.issubset({0, 1}):
        raise RuntimeError(
            f"Unexpected binary labels: "
            f"{unique_labels}"
        )

    print("Binary label validation: PASS")

    # -----------------------------------------------------------------------
    # Attack-family validation
    # -----------------------------------------------------------------------

    valid_attack_families = {
        "benign",
        "gafgyt",
        "mirai",
    }

    observed_attack_families = set(
        train_sample.attack_families.tolist()
    )

    print(
        "Observed attack families: "
        f"{sorted(observed_attack_families)}"
    )

    if not observed_attack_families.issubset(
        valid_attack_families
    ):
        raise RuntimeError(
            "Unexpected attack-family values: "
            f"{observed_attack_families}"
        )

    print("Attack-family validation: PASS")

    # -----------------------------------------------------------------------
    # Test sample from a specific device
    # -----------------------------------------------------------------------

    print()
    print("Reading a test sample from Danmini_Doorbell...")
    print("-" * 70)

    test_sample = loader.sample(
        split="test",
        n_rows=10,
        device="Danmini_Doorbell",
    )

    print(
        f"Test sample feature shape: "
        f"{test_sample.features.shape}"
    )

    print(
        f"Test sample labels:        "
        f"{test_sample.binary_labels.tolist()}"
    )

    print(
        f"Test sample attack family: "
        f"{test_sample.attack_families.tolist()}"
    )

    print(
        f"Test sample source file:    "
        f"{test_sample.source_file}"
    )

    print(
        f"Test sample row numbers:    "
        f"{test_sample.row_numbers.tolist()}"
    )

    if test_sample.features.shape[1] != 115:
        raise RuntimeError(
            "Danmini test sample does not contain "
            "115 features"
        )

    print(
        "Device-specific test sample: PASS"
    )

    # -----------------------------------------------------------------------
    # Final result
    # -----------------------------------------------------------------------

    print()
    print("=" * 70)
    print("STAGE 5G.20D.3 COMPLETE")
    print("=" * 70)
    print()
    print(
        "The split-aware N-BaIoT data-access layer "
        "passed verification."
    )

    print()
    print("Verified:")
    print("  - Project import path")
    print("  - 89 source files")
    print("  - 7,062,606 total rows")
    print("  - Frozen train/validation/test counts")
    print("  - 9 logical IoT clients")
    print("  - Source split indexes")
    print("  - Training sample loading")
    print("  - 115 numerical features")
    print("  - Binary labels")
    print("  - Attack-family metadata")
    print("  - Device-specific test loading")
    print()


if __name__ == "__main__":
    main()