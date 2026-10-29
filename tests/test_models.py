from pathlib import Path

import pytest
import torch
import yaml

from parsel.models.factory import build_model, count_parameters


def test_unet_output_shape():
    model = build_model({"arch": "unet", "encoder": "resnet18", "encoder_weights": None}, 9)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 9, 64, 64)
    assert count_parameters(model) > 1_000_000


@pytest.mark.parametrize(
    "arch, encoder",
    [("fpn", "resnet18"), ("deeplabv3plus", "resnet18"), ("segformer", "mit_b0"),
     ("upernet", "resnet18")],
)  # fmt: skip
def test_other_architectures_build(arch, encoder):
    model = build_model({"arch": arch, "encoder": encoder, "encoder_weights": None}, 5)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 5, 64, 64)  # BN needs batch > 1


@pytest.mark.parametrize("path", sorted(Path("configs/train").glob("*.yaml")), ids=lambda p: p.stem)
def test_every_train_config_builds(path):
    cfg = yaml.safe_load(path.read_text())
    model = build_model({**cfg["model"], "encoder_weights": None}, 9)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 9, 64, 64)
    assert cfg["name"] == path.stem


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_model({"arch": "nope"}, 9)
