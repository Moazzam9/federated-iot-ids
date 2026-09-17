from pathlib import Path

import pytest
import torch

from src.data.nbaiot_loader import NBaIoTSplitLoader
from src.fl.client import train_client
from src.models.mlp import SmallMLP


def make_fake_loader_and_scaler():
    """
    Placeholder helper.

    Real N-BaIoT integration is tested separately because loading
    complete device training data is intentionally expensive.
    """
    return None, None


def test_train_client_rejects_empty_client_id():
    model = SmallMLP()

    with pytest.raises(ValueError):
        train_client(
            model=model,
            loader=None,
            scaler=None,
            client_id="",
        )


def test_train_client_rejects_invalid_epochs():
    model = SmallMLP()

    with pytest.raises(ValueError):
        train_client(
            model=model,
            loader=None,
            scaler=None,
            client_id="client_1",
            epochs=0,
        )


def test_train_client_rejects_invalid_batch_size():
    model = SmallMLP()

    with pytest.raises(ValueError):
        train_client(
            model=model,
            loader=None,
            scaler=None,
            client_id="client_1",
            batch_size=0,
        )


def test_train_client_rejects_invalid_learning_rate():
    model = SmallMLP()

    with pytest.raises(ValueError):
        train_client(
            model=model,
            loader=None,
            scaler=None,
            client_id="client_1",
            learning_rate=0.0,
        )


def test_train_client_rejects_non_cpu_device():
    model = SmallMLP()

    with pytest.raises(ValueError):
        train_client(
            model=model,
            loader=None,
            scaler=None,
            client_id="client_1",
            device="cuda",
        )