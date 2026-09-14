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
from src.data.preprocessing import EXPECTED_FEATURE_COUNT


SCALER_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "training_standard_scaler.pkl"
)

METADATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "preprocessing"
    / "training_standard_scaler_metadata.json"
)


def main() -> None:
    print("=" * 70)
    print("VERIFY SAVED N-BaIoT TRAINING SCALER")
    print("=" * 70)

    if not SCALER_PATH.exists():
        raise FileNotFoundError(
            f"Scaler artifact not found: {SCALER_PATH}"
        )

    if not METADATA_PATH.exists():
        raise FileNotFoundError(
            f"Scaler metadata not found: {METADATA_PATH}"
        )

    with SCALER_PATH.open("rb") as handle:
        fitted_scaler = pickle.load(handle)

    with METADATA_PATH.open(
        "r",
        encoding="utf-8",
    ) as handle:
        metadata = json.load(handle)

    print(f"Scaler loaded: {SCALER_PATH}")
    print(f"Metadata loaded: {METADATA_PATH}")

    if fitted_scaler.feature_count != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Scaler feature count mismatch."
        )

    if fitted_scaler.scaler.n_features_in_ != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Underlying scaler feature count mismatch."
        )

    if fitted_scaler.scaler.n_samples_seen_ != 4_943_824:
        raise RuntimeError(
            "Scaler training-row count mismatch."
        )

    if metadata["split_used_for_fitting"] != "train":
        raise RuntimeError(
            "Scaler metadata does not report train-only fitting."
        )

    if metadata["training_rows_used"] != 4_943_824:
        raise RuntimeError(
            "Metadata training-row count mismatch."
        )

    if metadata["validation_rows_used"] != 0:
        raise RuntimeError(
            "Validation rows were unexpectedly used."
        )

    if metadata["test_rows_used"] != 0:
        raise RuntimeError(
            "Test rows were unexpectedly used."
        )

    if metadata["feature_count"] != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Metadata feature count mismatch."
        )

    if not fitted_scaler.scaler.mean_.shape == (
        EXPECTED_FEATURE_COUNT,
    ):
        raise RuntimeError(
            "Scaler mean shape mismatch."
        )

    if not fitted_scaler.scaler.var_.shape == (
        EXPECTED_FEATURE_COUNT,
    ):
        raise RuntimeError(
            "Scaler variance shape mismatch."
        )

    print()
    print("Metadata checks: PASS")
    print("Feature-count checks: PASS")
    print("Training-row checks: PASS")
    print("Train-only checks: PASS")
    print("Scaler-statistics shape checks: PASS")

    if not (
        fitted_scaler.scaler.mean_
        == fitted_scaler.scaler.mean_
    ).all():
        raise RuntimeError(
            "Scaler means contain invalid values."
        )

    if not (
        fitted_scaler.scaler.var_
        == fitted_scaler.scaler.var_
    ).all():
        raise RuntimeError(
            "Scaler variances contain invalid values."
        )

    print("Scaler-statistics validity check: PASS")

    loader = NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
    )

    first_chunk = next(
        loader.iter_chunks(split="validation")
    )

    transformed = fitted_scaler.transform(
        first_chunk.features
    )

    if transformed.shape != first_chunk.features.shape:
        raise RuntimeError(
            "Transformed validation chunk shape mismatch."
        )

    if list(transformed.columns) != list(
        first_chunk.features.columns
    ):
        raise RuntimeError(
            "Feature columns changed during transformation."
        )

    import numpy as np

    if not np.isfinite(
        transformed.to_numpy()
    ).all():
        raise RuntimeError(
            "Transformed validation data contain "
            "non-finite values."
        )

    print("Validation transformation check: PASS")

    print()
    print("=" * 70)
    print("SAVED SCALER VERIFICATION: PASS")
    print("=" * 70)


if __name__ == "__main__":
    main()
