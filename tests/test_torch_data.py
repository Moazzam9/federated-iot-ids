import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

from src.data.preprocessing import (
    EXPECTED_FEATURE_COUNT,
    fit_training_scaler,
)
from src.data.torch_data import (
    BatchData,
    dataframe_to_tensors,
    make_dataloader,
    transform_chunk,
)


def make_features(rows: int) -> pd.DataFrame:
    """Create a deterministic 115-feature test DataFrame."""
    values = np.arange(
        rows * EXPECTED_FEATURE_COUNT,
        dtype=np.float64,
    ).reshape(
        rows,
        EXPECTED_FEATURE_COUNT,
    )

    columns = [
        f"feature_{index}"
        for index in range(EXPECTED_FEATURE_COUNT)
    ]

    return pd.DataFrame(
        values,
        columns=columns,
    )


def make_labels(rows: int) -> pd.Series:
    """Create deterministic binary labels."""
    values = np.arange(rows) % 2

    return pd.Series(
        values,
        name="label",
    )


def test_dataframe_to_tensors_shapes():
    features = make_features(10)
    labels = make_labels(10)

    x, y = dataframe_to_tensors(
        features,
        labels,
    )

    assert x.shape == (10, 115)
    assert y.shape == (10, 1)


def test_dataframe_to_tensors_dtypes():
    features = make_features(10)
    labels = make_labels(10)

    x, y = dataframe_to_tensors(
        features,
        labels,
    )

    assert x.dtype == torch.float32
    assert y.dtype == torch.float32


def test_dataframe_to_tensors_preserves_binary_labels():
    features = make_features(10)
    labels = make_labels(10)

    _, y = dataframe_to_tensors(
        features,
        labels,
    )

    assert torch.all(
        (y == 0.0) | (y == 1.0)
    )


def test_make_dataloader_returns_expected_type():
    features = make_features(20)
    labels = make_labels(20)

    loader = make_dataloader(
        features,
        labels,
        batch_size=8,
    )

    assert isinstance(
        loader,
        DataLoader,
    )


def test_make_dataloader_batch_sizes():
    features = make_features(20)
    labels = make_labels(20)

    loader = make_dataloader(
        features,
        labels,
        batch_size=8,
    )

    batches = list(loader)

    assert len(batches) == 3
    assert batches[0][0].shape == (8, 115)
    assert batches[1][0].shape == (8, 115)
    assert batches[2][0].shape == (4, 115)


def test_make_dataloader_preserves_labels():
    features = make_features(10)
    labels = make_labels(10)

    loader = make_dataloader(
        features,
        labels,
        batch_size=10,
    )

    _, batch_labels = next(
        iter(loader)
    )

    expected = torch.tensor(
        labels.to_numpy(),
        dtype=torch.float32,
    ).reshape(-1, 1)

    assert torch.equal(
        batch_labels,
        expected,
    )


def test_scaler_can_transform_before_tensor_conversion():
    features = make_features(20)
    labels = make_labels(20)

    scaler = fit_training_scaler(
        features
    )

    transformed = scaler.transform(
        features
    )

    x, y = dataframe_to_tensors(
        transformed,
        labels,
    )

    assert x.shape == (20, 115)
    assert y.shape == (20, 1)
    assert torch.isfinite(x).all()
    assert torch.isfinite(y).all()


def test_transform_chunk_returns_batch_data():
    features = make_features(20)
    labels = make_labels(20)

    scaler = fit_training_scaler(
        features
    )

    batches = list(
        transform_chunk(
            scaler=scaler,
            features=features,
            labels=labels,
            batch_size=8,
        )
    )

    assert len(batches) == 3
    assert all(
        isinstance(batch, BatchData)
        for batch in batches
    )


def test_transform_chunk_batch_shapes():
    features = make_features(20)
    labels = make_labels(20)

    scaler = fit_training_scaler(
        features
    )

    batches = list(
        transform_chunk(
            scaler=scaler,
            features=features,
            labels=labels,
            batch_size=8,
        )
    )

    assert batches[0].features.shape == (8, 115)
    assert batches[1].features.shape == (8, 115)
    assert batches[2].features.shape == (4, 115)

    assert batches[0].labels.shape == (8, 1)
    assert batches[1].labels.shape == (8, 1)
    assert batches[2].labels.shape == (4, 1)


def test_transform_chunk_outputs_finite_tensors():
    features = make_features(20)
    labels = make_labels(20)

    scaler = fit_training_scaler(
        features
    )

    batches = list(
        transform_chunk(
            scaler=scaler,
            features=features,
            labels=labels,
            batch_size=8,
        )
    )

    for batch in batches:
        assert torch.isfinite(
            batch.features
        ).all()

        assert torch.isfinite(
            batch.labels
        ).all()


def test_transform_chunk_does_not_refit_scaler():
    training = make_features(20)
    other_data = make_features(10) + 100000.0
    labels = make_labels(10)

    scaler = fit_training_scaler(
        training
    )

    original_mean = scaler.scaler.mean_.copy()
    original_scale = scaler.scaler.scale_.copy()

    list(
        transform_chunk(
            scaler=scaler,
            features=other_data,
            labels=labels,
            batch_size=8,
        )
    )

    assert np.array_equal(
        scaler.scaler.mean_,
        original_mean,
    )

    assert np.array_equal(
        scaler.scaler.scale_,
        original_scale,
    )


def test_wrong_feature_count_is_rejected():
    features = pd.DataFrame(
        np.ones(
            (
                10,
                EXPECTED_FEATURE_COUNT - 1,
            )
        )
    )

    labels = make_labels(10)

    with pytest.raises(
        ValueError,
        match="Expected 115 features",
    ):
        dataframe_to_tensors(
            features,
            labels,
        )


def test_mismatched_row_count_is_rejected():
    features = make_features(10)
    labels = make_labels(9)

    with pytest.raises(
        ValueError,
        match="same number of rows",
    ):
        dataframe_to_tensors(
            features,
            labels,
        )


def test_invalid_batch_size_is_rejected():
    features = make_features(10)
    labels = make_labels(10)

    with pytest.raises(
        ValueError,
        match="batch_size must be greater than zero",
    ):
        make_dataloader(
            features,
            labels,
            batch_size=0,
        )


def test_batch_data_container():
    features = torch.randn(
        4,
        115,
    )

    labels = torch.randint(
        0,
        2,
        (4, 1),
    ).float()

    batch = BatchData(
        features=features,
        labels=labels,
    )

    assert batch.features.shape == (4, 115)
    assert batch.labels.shape == (4, 1)