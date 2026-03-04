import torch
import timm
from torch import Tensor, nn


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str) -> None:
        super().__init__()
        self.backbone_name = backbone
        self.backbone = timm.create_model(
            model_name=backbone,
            pretrained=True,
            num_classes=0,
            global_pool="avg",
        )

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return features


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
