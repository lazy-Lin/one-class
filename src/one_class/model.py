import torch
import timm
from torch import Tensor, nn


class FeatureExtractor(nn.Module):
    def __init__(self, backbone: str, pretrained_weights: str | None = None) -> None:
        super().__init__()
        self.backbone_name = backbone

        # 如果提供了本地权重路径，则先不加载预训练权重（pretrained=False）
        # 否则默认加载 timm 在线权重（pretrained=True）
        use_pretrained = pretrained_weights is None

        self.backbone = timm.create_model(
            model_name=backbone,
            pretrained=use_pretrained,
            num_classes=0,
            global_pool="avg",
        )

        if pretrained_weights:
            self.load_local_weights(pretrained_weights)

    def load_local_weights(self, path: str) -> None:
        state_dict = torch.load(path, map_location="cpu")
        # 兼容 timm 的权重 key（有些可能带前缀，视来源而定）
        # 这里假设是标准的 backbone 权重或完整模型权重
        # 如果 key 不匹配，通常会报错，这里不做复杂处理，直接 load
        # 也可以尝试 strict=False
        try:
            self.backbone.load_state_dict(state_dict, strict=True)
            print(f"Successfully loaded local weights from: {path}")
        except RuntimeError:
            print(f"[Warning] Strict loading failed for {path}. Trying strict=False...")
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            if missing:
                print(f"[Warning] Missing keys: {missing[:5]} ...")
            if unexpected:
                print(f"[Warning] Unexpected keys: {unexpected[:5]} ...")
            print(f"Loaded local weights (loose match) from: {path}")

    def forward(self, x: Tensor) -> Tensor:
        features = self.backbone(x)
        if features.ndim == 1:
            features = features.unsqueeze(0)
        return features


def resolve_device(device: str | None) -> torch.device:
    if device:
        return torch.device(device)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")
