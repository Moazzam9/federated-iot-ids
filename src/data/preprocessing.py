from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


EXPECTED_FEATURE_COUNT = 115


@dataclass
class FittedStandardScaler:
    """Training-fitted StandardScaler for the N-BaIoT features."""

    scaler: StandardScaler
    feature_count: int = EXPECTED_FEATURE_COUNT

    @classmethod
    def fit(cls, features: pd.DataFrame) -> "FittedStandardScaler":
        """
        Fit the scaler using training features only.

        This method must only be called with training data.
        """
        _validate_features(features)

        scaler = StandardScaler()
        scaler.fit(features)

        return cls(
            scaler=scaler,
            feature_count=features.shape[1],
        )

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

    if not np.isfinite(features.to_numpy(dtype=np.float64)).all():
        raise ValueError("Features contain NaN or infinite values.")


def fit_training_scaler(
    training_features: pd.DataFrame,
) -> FittedStandardScaler:
    """
    Fit a StandardScaler on training features.

    This function intentionally accepts only the training feature matrix.
    Validation and test data must be passed to transform_features() instead.
    """
    return FittedStandardScaler.fit(training_features)


def transform_features(
    scaler: FittedStandardScaler,
    features: pd.DataFrame,
) -> pd.DataFrame:
    """Transform a feature matrix using a previously fitted scaler."""
    if not isinstance(scaler, FittedStandardScaler):
        raise TypeError(
            "scaler must be a FittedStandardScaler returned by "
            "fit_training_scaler()."
        )

    return scaler.transform(features)