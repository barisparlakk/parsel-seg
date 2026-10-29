import pytest
import torch

from parsel.models.factory import build_model, count_parameters


def test_unet_output_shape():
    model = build_model({"arch": "unet", "encoder": "resnet18", "encoder_weights": None}, 9)
    out = model(torch.randn(2, 3, 64, 64))
    assert out.shape == (2, 9, 64, 64)
    assert count_parameters(model) > 1_000_000


@pytest.mark.parametrize("arch", ["fpn", "deeplabv3plus"])
def test_other_architectures_build(arch):
    model = build_model({"arch": arch, "encoder": "resnet18", "encoder_weights": None}, 5)
    assert model(torch.randn(2, 3, 64, 64)).shape == (2, 5, 64, 64)  # BN needs batch > 1


def test_unknown_arch_raises():
    with pytest.raises(ValueError):
        build_model({"arch": "nope"}, 9)
