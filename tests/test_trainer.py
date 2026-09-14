import numpy as np
import torch
from torch import nn

from src.models.mlp import SmallMLP
from src.models.trainer import (
    evaluate_batches,
    set_random_seed,
    train_one_epoch,
)


def make_batches(
    rows: int = 32,
    batch_size: int = 8,
):
    features = torch.randn(
        rows,
        115,
        dtype=torch.float32,
    )

    labels = torch.randint(
        0,
        2,
        (rows, 1),
        dtype=torch.float32,
    )

    for start in range(
        0,
        rows,
        batch_size,
    ):
        end = min(
            start + batch_size,
            rows,
        )

        yield (
            features[start:end],
            labels[start:end],
        )


def test_random_seed_is_reproducible() -> None:
    set_random_seed(42)

    first = torch.rand(10)

    set_random_seed(42)

    second = torch.rand(10)

    assert torch.equal(first, second)


def test_train_one_epoch_updates_model() -> None:
    set_random_seed(42)

    model = SmallMLP()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    criterion = nn.BCELoss()

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    metrics = train_one_epoch(
        model=model,
        batches=make_batches(),
        optimizer=optimizer,
        criterion=criterion,
    )

    after = list(model.parameters())

    assert metrics.samples == 32
    assert metrics.loss >= 0.0
    assert metrics.elapsed_seconds >= 0.0

    changed = any(
        not torch.equal(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(
            before,
            after,
        )
    )

    assert changed


def test_train_one_epoch_rejects_empty_batches() -> None:
    model = SmallMLP()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=0.001,
    )

    criterion = nn.BCELoss()

    empty_batches = iter(())

    try:
        train_one_epoch(
            model=model,
            batches=empty_batches,
            optimizer=optimizer,
            criterion=criterion,
        )
    except ValueError as error:
        assert "No samples" in str(error)
    else:
        raise AssertionError(
            "Expected train_one_epoch() to reject empty batches."
        )


def test_evaluation_returns_valid_metrics() -> None:
    set_random_seed(42)

    model = SmallMLP()

    criterion = nn.BCELoss()

    metrics = evaluate_batches(
        model=model,
        batches=make_batches(
            rows=40,
            batch_size=10,
        ),
        criterion=criterion,
    )

    assert metrics.samples == 40
    assert metrics.loss >= 0.0

    assert 0.0 <= metrics.accuracy <= 1.0
    assert 0.0 <= metrics.precision <= 1.0
    assert 0.0 <= metrics.recall <= 1.0
    assert 0.0 <= metrics.f1 <= 1.0
    assert 0.0 <= metrics.roc_auc <= 1.0


def test_evaluation_matches_perfect_predictions() -> None:
    class PerfectModel(nn.Module):
        def forward(
            self,
            features: torch.Tensor,
        ) -> torch.Tensor:
            return features[:, :1]

    model = PerfectModel()

    features = torch.tensor(
        [
            [0.0],
            [1.0],
            [0.0],
            [1.0],
        ],
        dtype=torch.float32,
    )

    features = torch.cat(
        [
            features,
            torch.zeros(
                4,
                114,
                dtype=torch.float32,
            ),
        ],
        dim=1,
    )

    labels = torch.tensor(
        [
            [0.0],
            [1.0],
            [0.0],
            [1.0],
        ],
        dtype=torch.float32,
    )

    metrics = evaluate_batches(
        model=model,
        batches=iter(
            [
                (
                    features,
                    labels,
                )
            ]
        ),
        criterion=nn.BCELoss(),
    )

    assert metrics.accuracy == 1.0
    assert metrics.precision == 1.0
    assert metrics.recall == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0


def test_evaluation_rejects_empty_batches() -> None:
    model = SmallMLP()

    criterion = nn.BCELoss()

    try:
        evaluate_batches(
            model=model,
            batches=iter(()),
            criterion=criterion,
        )
    except ValueError as error:
        assert "No samples" in str(error)
    else:
        raise AssertionError(
            "Expected evaluate_batches() to reject empty batches."
        )
