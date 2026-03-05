from dataclasses import dataclass


@dataclass
class TrainConfig:
    train_dir: str = "data/train/normal"
    val_dir: str = "data/val/normal"
    output_dir: str = "artifacts/faiss_patchcore"
    backbone: str = "resnet18"
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    knn_k: int = 5
    threshold_quantile: float = 0.995
    device: str | None = None
    pretrained_weights: str | None = None


CONFIG = TrainConfig()
