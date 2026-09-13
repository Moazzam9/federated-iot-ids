import torch

from src.models.mlp import SmallMLP, count_trainable_parameters


def test_default_model_parameter_count():
    model = SmallMLP()

    assert count_trainable_parameters(model) == 9537


def test_model_accepts_115_features():
    model = SmallMLP()

    x = torch.randn(8, 115)
    y = model(x)

    assert y.shape == (8, 1)


def test_model_outputs_probabilities():
    model = SmallMLP()

    x = torch.randn(16, 115)
    y = model(x)

    assert torch.all(y >= 0.0)
    assert torch.all(y <= 1.0)


def test_model_contains_expected_layers():
    model = SmallMLP()

    layers = list(model.network)

    assert isinstance(layers[0], torch.nn.Linear)
    assert layers[0].in_features == 115
    assert layers[0].out_features == 64

    assert isinstance(layers[1], torch.nn.ReLU)

    assert isinstance(layers[2], torch.nn.Linear)
    assert layers[2].in_features == 64
    assert layers[2].out_features == 32

    assert isinstance(layers[3], torch.nn.ReLU)

    assert isinstance(layers[4], torch.nn.Linear)
    assert layers[4].in_features == 32
    assert layers[4].out_features == 1

    assert isinstance(layers[5], torch.nn.Sigmoid)


def test_model_produces_finite_outputs():
    model = SmallMLP()

    x = torch.randn(32, 115)
    y = model(x)

    assert torch.isfinite(y).all()


def test_model_supports_backward_pass():
    model = SmallMLP()

    x = torch.randn(8, 115)
    y = model(x)

    targets = torch.randint(0, 2, (8, 1)).float()

    loss_function = torch.nn.BCELoss()
    loss = loss_function(y, targets)

    loss.backward()

    assert torch.isfinite(loss).item()

    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
