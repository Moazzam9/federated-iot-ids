from __future__ import annotations

from pathlib import Path
import yaml
import pytest

from src.models.mlp import SmallMLP, count_trainable_parameters

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_reproducibility_config_exists():
    """Verify that experiment.yaml configuration exists and is valid YAML."""
    config_path = PROJECT_ROOT / "configs" / "experiment.yaml"
    assert config_path.is_file(), f"Missing configuration file: {config_path}"

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    assert config["project"]["name"] == "federated-iot-ids"
    assert config["dataset"]["name"] == "N-BaIoT"
    assert config["model"]["name"] == "small_mlp"
    assert config["reproducibility"]["seed"] == 42


def test_reproducibility_split_specification_exists():
    """Verify that frozen split_specification.txt exists."""
    split_spec_path = PROJECT_ROOT / "data" / "processed" / "splits" / "split_specification.txt"
    assert split_spec_path.is_file(), f"Missing split specification: {split_spec_path}"

    content = split_spec_path.read_text(encoding="utf-8")
    assert "N-BaIoT DATASET SPLIT SPECIFICATION" in content
    assert "7,062,606" in content


def test_reproducibility_scaler_binary_exists():
    """Verify that pre-fitted training standard scaler pickle exists."""
    scaler_path = PROJECT_ROOT / "data" / "processed" / "preprocessing" / "training_standard_scaler.pkl"
    assert scaler_path.is_file(), f"Missing standard scaler binary: {scaler_path}"


def test_reproducibility_guide_exists():
    """Verify that reproducibility.md documentation exists."""
    repro_doc = PROJECT_ROOT / "docs" / "experiments" / "reproducibility.md"
    assert repro_doc.is_file(), f"Missing reproducibility guide: {repro_doc}"


def test_model_parameter_count_matches_study():
    """Verify that SmallMLP parameter count is exactly 9,537."""
    model = SmallMLP(input_dim=115, hidden_dim_1=64, hidden_dim_2=32)
    param_count = count_trainable_parameters(model)
    assert param_count == 9537, f"Expected 9,537 parameters, got {param_count}"
