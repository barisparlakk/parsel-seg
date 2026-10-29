"""Model factory: builds a segmentation network from a config dict."""

import segmentation_models_pytorch as smp
from torch import nn

ARCHS = {
    "unet": smp.Unet,
    "unetplusplus": smp.UnetPlusPlus,
    "deeplabv3plus": smp.DeepLabV3Plus,
    "fpn": smp.FPN,
    "segformer": smp.Segformer,
    "upernet": smp.UPerNet,
}


def build_model(cfg: dict, num_classes: int) -> nn.Module:
    """Build a model from `cfg` with keys arch, encoder and encoder_weights.

    encoder_weights is "imagenet" (default) or null for random initialisation.
    """
    arch = cfg.get("arch", "unet").lower()
    if arch not in ARCHS:
        raise ValueError(f"Unknown arch {arch!r}; choose from {sorted(ARCHS)}")
    return ARCHS[arch](
        encoder_name=cfg.get("encoder", "resnet34"),
        encoder_weights=cfg.get("encoder_weights", "imagenet"),
        in_channels=3,
        classes=num_classes,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
