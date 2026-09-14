import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import (
    EXPECTED_FEATURE_COUNT,
    FittedStandardScaler,
    fit_training_scaler,
    transform_features,
)


def make_features(rows: int, start: float = 0.0) -> pd.DataFrame:
    """Create a deterministic 115-feature test DataFrame."""
    values = np.arange(
        start,
        start + rows * EXPECTED_FEATURE_COUNT,
        dtype=np.float64,
    ).reshape(rows, EXPECTED_FEATURE_COUNT)

    columns = [f"feature_{index}" for index in range(EXPECTED_FEATURE_COUNT)]

    return pd.DataFrame(values, columns=columns)


def test_scaler_fits_on_training_data():
    training = make_features(rows=20)

    scaler = fit_training_scaler(training)

    assert isinstance(scaler, FittedStandardScaler)
    assert scaler.feature_count == EXPECTED_FEATURE_COUNT


def test_training_data_are_standardized():
    training = make_features(rows=20)

    scaler = fit_training_scaler(training)
    transformed = transform_features(scaler, training)

    means = transformed.mean(axis=0).to_numpy()
    standard_deviations = transformed.std(axis=0, ddof=0).to_numpy()

    assert np.allclose(means, 0.0, atol=1e-10)
    assert np.allclose(standard_deviations, 1.0, atol=1e-10)


def test_validation_data_are_transformed_without_refitting():
    training = make_features(rows=20, start=0.0)
    validation = make_features(rows=10, start=100000.0)

    scaler = fit_training_scaler(training)

    original_mean = scaler.scaler.mean_.copy()
    original_scale = scaler.scaler.scale_.copy()

    transformed_validation = transform_features(scaler, validation)

    assert transformed_validation.shape == validation.shape
    assert np.array_equal(scaler.scaler.mean_, original_mean)
    assert np.array_equal(scaler.scaler.scale_, original_scale)


def test_test_data_are_transformed_without_refitting():
    training = make_features(rows=20, start=0.0)
    test = make_features(rows=10, start=200000.0)

    scaler = fit_training_scaler(training)

    original_mean = scaler.scaler.mean_.copy()
    original_scale = scaler.scaler.scale_.copy()

    transformed_test = transform_features(scaler, test)

    assert transformed_test.shape == test.shape
    assert np.array_equal(scaler.scaler.mean_, original_mean)
    assert np.array_equal(scaler.scaler.scale_, original_scale)


def test_wrong_feature_count_is_rejected():
    training = make_features(rows=10)

    scaler = fit_training_scaler(training)

    invalid_features = pd.DataFrame(
        np.ones((5, EXPECTED_FEATURE_COUNT - 1))
    )

    with pytest.raises(ValueError, match="Expected 115 features"):
        transform_features(scaler, invalid_features)


def test_non_finite_values_are_rejected():
    training = make_features(rows=10)
    training.iloc[0, 0] = np.nan

    with pytest.raises(ValueError, match="NaN or infinite"):
        fit_training_scaler(training)


def test_scaler_preserves_feature_columns():
    training = make_features(rows=10)

    scaler = fit_training_scaler(training)
    transformed = transform_features(scaler, training)

    assert list(transformed.columns) == list(training.columns)


def test_scaler_preserves_row_index():
    training = make_features(rows=10)
    training.index = range(100, 110)

    scaler = fit_training_scaler(training)
    transformed = transform_features(scaler, training)

    assert list(transformed.index) == list(training.index)