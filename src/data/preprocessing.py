from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

if TYPE_CHECKING:
    from src.data.nbaiot_loader import NBaIoTSplitLoader


EXPECTED_FEATURE_COUNT = 115


@dataclass
class FittedStandardScaler:
    """Training-fitted StandardScaler for the N-BaIoT features."""

    scaler: StandardScaler
    feature_count: int = EXPECTED_FEATURE_COUNT

    @classmethod
    def fit(cls, features: pd.DataFrame) -> "FittedStandardScaler":
        """
        Fit the scaler using an in-memory training feature matrix.

        This method must only be called with training data.
        """
        _validate_features(features)

        scaler = StandardScaler()
        scaler.fit(features)

        return cls(
            scaler=scaler,
            feature_count=features.shape[1],
        )

    @classmethod
    def partial_fit(
        cls,
        scaler: StandardScaler | None,
        features: pd.DataFrame,
    ) -> StandardScaler:
        """
        Incrementally fit a StandardScaler on one training chunk.

        This method is intended for streaming large training datasets.
        It does not retain the complete training dataset in memory.
        """
        _validate_features(features)

        if scaler is None:
            scaler = StandardScaler()

        scaler.partial_fit(features)

        return scaler

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Transform features using the already-fitted training scaler."""
        _validate_features(features)

        transformed = self.scaler.transform(features)

        return pd.DataFrame(
            transformed,
            columns=features.columns,
            index=features.index,
        )


def _validate_features(features: pd.DataFrame) -> None:
    """Validate the feature matrix before fitting or transforming."""
    if not isinstance(features, pd.DataFrame):
        raise TypeError("features must be a pandas DataFrame.")

    if features.shape[1] != EXPECTED_FEATURE_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_FEATURE_COUNT} features, "
            f"but received {features.shape[1]}."
        )

    if not np.isfinite(
        features.to_numpy(dtype=np.float64)
    ).all():
        raise ValueError(
            "Features contain NaN or infinite values."
        )


def fit_training_scaler(
    training_features: pd.DataFrame,
) -> FittedStandardScaler:
    """
    Fit a StandardScaler on an in-memory training feature matrix.

    This function intentionally accepts only training features.
    Validation and test data must be transformed, never fitted.
    """
    return FittedStandardScaler.fit(training_features)


def fit_training_scaler_incremental(
    loader: "NBaIoTSplitLoader",
) -> FittedStandardScaler:
    """
    Fit a StandardScaler over the complete N-BaIoT training split
    incrementally, one loader chunk at a time.

    Only training rows are used. Validation and test rows are never
    passed to the scaler fitting process.
    """
    scaler: StandardScaler | None = None
    total_rows = 0

    for data_chunk in loader.iter_chunks(
        split="train",
    ):
        features = data_chunk.features

        _validate_features(features)

        scaler = FittedStandardScaler.partial_fit(
            scaler=scaler,
            features=features,
        )

        total_rows += len(features)

    if scaler is None or total_rows == 0:
        raise ValueError(
            "No training rows were available for incremental "
            "scaler fitting."
        )

    if not hasattr(scaler, "n_features_in_"):
        raise RuntimeError(
            "Incremental scaler fitting completed without "
            "fitted feature statistics."
        )

    if scaler.n_features_in_ != EXPECTED_FEATURE_COUNT:
        raise RuntimeError(
            "Incremental scaler has an unexpected feature count.\n"
            f"Expected: {EXPECTED_FEATURE_COUNT}\n"
            f"Actual: {scaler.n_features_in_}"
        )

    if not np.isfinite(scaler.mean_).all():
        raise RuntimeError(
            "Incremental scaler contains non-finite mean values."
        )

    if not np.isfinite(scaler.var_).all():
        raise RuntimeError(
            "Incremental scaler contains non-finite variance values."
        )

    return FittedStandardScaler(
        scaler=scaler,
        feature_count=EXPECTED_FEATURE_COUNT,
    )


def transform_features(
    scaler: FittedStandardScaler,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Transform a feature matrix using a previously fitted scaler."""
    if not isinstance(scaler, FittedStandardScaler):
        raise TypeError(
            "scaler must be a FittedStandardScaler returned by "
            "fit_training_scaler() or "
            "fit_training_scaler_incremental()."
        )

    return scaler.transform(features)
