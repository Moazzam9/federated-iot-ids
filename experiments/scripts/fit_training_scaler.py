from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path


# ----------------------------------------------------------------------
# Make the project root importable when this file is executed directly.
# ----------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import fit_training_scaler_incremental


EXPECTED_TRAIN_ROWS = 4_943_824
EXPECTED_FEATURES = 115

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed" / "preprocessing"
SCALER_PATH = OUTPUT_DIR / "training_standard_scaler.pkl"
METADATA_PATH = OUTPUT_DIR / "training_standard_scaler_metadata.json"


def main() -> None:
    print("=" * 70)
    print("N-BaIoT TRAINING-ONLY STANDARD SCALER FIT")
    print("=" * 70)

    print(f"Project root: {PROJECT_ROOT}")
    print(f"Expected training rows: {EXPECTED_TRAIN_ROWS:,}")
    print(f"Expected features: {EXPECTED_FEATURES}")

    loader = NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
    )

    print()
    print("Validating source split indexes...")
    loader.validate_source_indexes()
    print("Source split indexes: PASS")

    print()
    print("Fitting StandardScaler incrementally...")
    print("Only split='train' will be used.")
    print("Validation and test rows will NOT be used.")
    print()

    scaler = fit_training_scaler_incremental(loader)

    actual_rows = int(scaler.scaler.n_samples_seen_)
    actual_features = int(scaler.scaler.n_features_in_)

    print()
    print("Scaler fitting completed.")
    print(f"Rows used: {actual_rows:,}")
    print(f"Features: {actual_features}")

    if actual_rows != EXPECTED_TRAIN_ROWS:
        raise RuntimeError(
            "Training-row count mismatch.\n"
            f"Expected: {EXPECTED_TRAIN_ROWS:,}\n"
            f"Actual:   {actual_rows:,}"
        )

    if actual_features != EXPECTED_FEATURES:
        raise RuntimeError(
            "Feature-count mismatch.\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Actual: {actual_features}"
        )

    if scaler.feature_count != EXPECTED_FEATURES:
        raise RuntimeError(
            "Wrapped scaler feature-count mismatch.\n"
            f"Expected: {EXPECTED_FEATURES}\n"
            f"Actual: {scaler.feature_count}"
        )

    if scaler.scaler.mean_.shape != (EXPECTED_FEATURES,):
        raise RuntimeError("Unexpected scaler mean shape.")

    if scaler.scaler.var_.shape != (EXPECTED_FEATURES,):
        raise RuntimeError("Unexpected scaler variance shape.")

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    with SCALER_PATH.open("wb") as handle:
        pickle.dump(
            scaler,
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )

    metadata = {
        "dataset": "N-BaIoT",
        "split_used_for_fitting": "train",
        "training_rows_used": actual_rows,
        "expected_training_rows": EXPECTED_TRAIN_ROWS,
        "feature_count": actual_features,
        "expected_feature_count": EXPECTED_FEATURES,
        "method": "StandardScaler",
        "fit_mode": "incremental_partial_fit",
        "validation_rows_used": 0,
        "test_rows_used": 0,
        "scaler_path": str(
            SCALER_PATH.relative_to(PROJECT_ROOT)
        ),
        "purpose": (
            "Training-only preprocessing artifact for centralized, "
            "local-only, and federated experiments."
        ),
    }

    with METADATA_PATH.open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(
            metadata,
            handle,
            indent=2,
        )
        handle.write("\n")

    print()
    print("=" * 70)
    print("SCALER ARTIFACT VERIFICATION")
    print("=" * 70)
    print("Training rows: PASS")
    print("Feature count: PASS")
    print("Validation rows used: 0")
    print("Test rows used: 0")
    print()
    print(f"Scaler:   {SCALER_PATH}")
    print(f"Metadata: {METADATA_PATH}")
    print()
    print("STAGE 5G.20F.5: PASS")


if __name__ == "__main__":
    main()
