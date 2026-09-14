from pathlib import Path

import numpy as np
import pytest

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.data.preprocessing import fit_training_scaler
from src.data.torch_data import iter_torch_batches


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT.parent / "federated-iot-temp"


@pytest.fixture(scope="module")
def nbaiot_loader() -> NBaIoTSplitLoader:
    if not DATA_ROOT.exists():
        pytest.skip(
            f"N-BaIoT data root does not exist: {DATA_ROOT}"
        )

    return NBaIoTSplitLoader(
        project_root=PROJECT_ROOT,
        data_root=DATA_ROOT,
        chunk_size=5_000,
    )


def test_real_nbaiot_training_chunk_can_fit_scaler(
    nbaiot_loader: NBaIoTSplitLoader,
) -> None:
    """
    Read a real training chunk from N-BaIoT and fit the scaler only
    on training data.
    """
    chunks = nbaiot_loader.iter_chunks(
        split="train",
    )

    first_chunk = next(chunks)

    assert first_chunk.split == "train"
    assert first_chunk.features.shape[1] == 115
    assert len(first_chunk.features) == len(
        first_chunk.binary_labels
    )

    assert set(
        first_chunk.binary_labels.unique()
    ).issubset({0, 1})

    scaler = fit_training_scaler(
        first_chunk.features
    )

    transformed = scaler.transform(
        first_chunk.features
    )

    assert transformed.shape == first_chunk.features.shape
    assert np.isfinite(transformed.to_numpy()).all()


def test_real_nbaiot_chunk_reaches_pytorch(
    nbaiot_loader: NBaIoTSplitLoader,
) -> None:
    """
    Verify the complete real-data path:

    N-BaIoT loader
    -> frozen split
    -> preprocessing
    -> NumPy float32
    -> PyTorch tensors
    -> DataLoader batches
    """
    train_chunks = nbaiot_loader.iter_chunks(
        split="train",
    )

    first_train_chunk = next(train_chunks)

    scaler = fit_training_scaler(
        first_train_chunk.features
    )

    batches = iter_torch_batches(
        loader=nbaiot_loader,
        scaler=scaler,
        split="train",
        batch_size=256,
    )

    first_batch = next(batches)

    assert first_batch.features.ndim == 2
    assert first_batch.features.shape[1] == 115

    assert first_batch.labels.ndim == 2
    assert first_batch.labels.shape[1] == 1

    assert first_batch.features.shape[0] <= 256

    assert str(first_batch.features.dtype) == "torch.float32"
    assert str(first_batch.labels.dtype) == "torch.float32"

    assert np.isfinite(
        first_batch.features.numpy()
    ).all()

    assert set(
        first_batch.labels.numpy().reshape(-1).tolist()
    ).issubset({0.0, 1.0})


def test_real_nbaiot_validation_split_reaches_pytorch(
    nbaiot_loader: NBaIoTSplitLoader,
) -> None:
    """
    Verify that validation data can be streamed through the same
    training-fitted preprocessing boundary without refitting.
    """
    train_chunks = nbaiot_loader.iter_chunks(
        split="train",
    )

    first_train_chunk = next(train_chunks)

    scaler = fit_training_scaler(
        first_train_chunk.features
    )

    validation_batches = iter_torch_batches(
        loader=nbaiot_loader,
        scaler=scaler,
        split="validation",
        batch_size=128,
    )

    first_batch = next(validation_batches)

    assert first_batch.features.shape[1] == 115
    assert first_batch.labels.shape[1] == 1
    assert first_batch.features.shape[0] <= 128

    assert str(first_batch.features.dtype) == "torch.float32"
    assert str(first_batch.labels.dtype) == "torch.float32"

    assert np.isfinite(
        first_batch.features.numpy()
    ).all()

    assert set(
        first_batch.labels.numpy().reshape(-1).tolist()
    ).issubset({0.0, 1.0})
