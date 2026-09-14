import numpy as np
import pandas as pd
import pytest
from sklearn.preprocessing import StandardScaler

from src.data.preprocessing import (
    EXPECTED_FEATURE_COUNT,
    FittedStandardScaler,
    fit_training_scaler,
)


def make_features(
    rows: int,
    start: float = 0.0,
) -> pd.DataFrame:
    values = np.arange(
        start,
        start + rows * EXPECTED_FEATURE_COUNT,
        dtype=np.float64,
    ).reshape(rows, EXPECTED_FEATURE_COUNT)

    return pd.DataFrame(values)


def test_partial_fit_matches_standard_scaler() -> None:
    """
    Incremental fitting over multiple chunks should produce the same
    population statistics as fitting StandardScaler on the complete
    training matrix.
    """
    chunk_1 = make_features(4, start=0.0)
    chunk_2 = make_features(
        6,
        start=4 * EXPECTED_FEATURE_COUNT,
    )

    complete = pd.concat(
        [chunk_1, chunk_2],
        ignore_index=True,
    )

    expected = StandardScaler()
    expected.fit(complete)

    incremental = None

    incremental = FittedStandardScaler.partial_fit(
        incremental,
        chunk_1,
    )

    incremental = FittedStandardScaler.partial_fit(
        incremental,
        chunk_2,
    )

    assert incremental is not None
    assert incremental.n_features_in_ == EXPECTED_FEATURE_COUNT

    np.testing.assert_allclose(
        incremental.mean_,
        expected.mean_,
        rtol=1e-10,
        atol=1e-10,
    )

    np.testing.assert_allclose(
        incremental.var_,
        expected.var_,
        rtol=1e-10,
        atol=1e-10,
    )


def test_incremental_scaler_can_transform() -> None:
    chunk_1 = make_features(5, start=0.0)
    chunk_2 = make_features(
        5,
        start=5 * EXPECTED_FEATURE_COUNT,
    )

    scaler = None

    scaler = FittedStandardScaler.partial_fit(
        scaler,
        chunk_1,
    )

    scaler = FittedStandardScaler.partial_fit(
        scaler,
        chunk_2,
    )

    wrapped = FittedStandardScaler(
        scaler=scaler,
    )

    transformed = wrapped.transform(chunk_2)

    assert transformed.shape == chunk_2.shape
    assert list(transformed.columns) == list(
        chunk_2.columns
    )

    assert np.isfinite(
        transformed.to_numpy()
    ).all()


def test_incremental_fit_rejects_wrong_feature_count() -> None:
    bad_features = pd.DataFrame(
        np.zeros((5, EXPECTED_FEATURE_COUNT - 1))
    )

    with pytest.raises(ValueError, match="Expected 115 features"):
        FittedStandardScaler.partial_fit(
            None,
            bad_features,
        )


def test_incremental_fit_rejects_non_finite_values() -> None:
    features = make_features(5)
    features.iloc[0, 0] = np.nan

    with pytest.raises(
        ValueError,
        match="NaN or infinite",
    ):
        FittedStandardScaler.partial_fit(
            None,
            features,
        )


def test_incremental_fit_requires_training_chunks() -> None:
    """
    This test documents the API contract: the incremental function
    receives a loader and itself requests only the train split.
    """
    from src.data.preprocessing import (
        fit_training_scaler_incremental,
    )

    class FakeChunk:
        def __init__(self, features: pd.DataFrame) -> None:
            self.features = features

    class FakeLoader:
        def __init__(self) -> None:
            self.requested_splits = []

        def iter_chunks(self, split: str):
            self.requested_splits.append(split)

            if split != "train":
                raise AssertionError(
                    "Incremental scaler requested a non-training split."
                )

            yield FakeChunk(make_features(3))

    loader = FakeLoader()

    scaler = fit_training_scaler_incremental(loader)

    assert loader.requested_splits == ["train"]
    assert scaler.feature_count == EXPECTED_FEATURE_COUNT
    assert scaler.scaler.n_features_in_ == EXPECTED_FEATURE_COUNT
