import json
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from .data import list_images, make_loader
from .model import FeatureExtractor, resolve_device


@dataclass
class PipelineConfig:
    train_dir: str
    val_dir: str
    output_dir: str
    backbone: str = "resnet18"
    image_size: int = 224
    batch_size: int = 64
    num_workers: int = 4
    knn_k: int = 5
    threshold_quantile: float = 0.995
    device: str | None = None
    pretrained_weights: str | None = None


class OneClassFaissPipeline:
    def __init__(self, config: PipelineConfig) -> None:
        self.config = config
        self.device = resolve_device(config.device)
        self.model = FeatureExtractor(
            backbone=config.backbone,
            pretrained_weights=config.pretrained_weights
        ).to(self.device).eval()

    def run(self) -> dict[str, float | str | int]:
        train_paths = list_images(Path(self.config.train_dir))
        val_paths = list_images(Path(self.config.val_dir))
        if not train_paths:
            raise ValueError("train_dir 下没有可用图像")
        if not val_paths:
            raise ValueError("val_dir 下没有可用图像")
        train_loader = make_loader(
            image_paths=train_paths,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        val_loader = make_loader(
            image_paths=val_paths,
            image_size=self.config.image_size,
            batch_size=self.config.batch_size,
            num_workers=self.config.num_workers,
            shuffle=False,
        )
        train_embeddings = self._extract_embeddings(train_loader)
        val_embeddings = self._extract_embeddings(val_loader)
        index = self._build_index(train_embeddings)
        val_scores = self._knn_score(index, val_embeddings, self.config.knn_k)
        threshold = float(np.quantile(val_scores, self.config.threshold_quantile))
        metrics = {
            "train_samples": int(len(train_paths)),
            "val_samples": int(len(val_paths)),
            "embedding_dim": int(train_embeddings.shape[1]),
            "threshold": threshold,
            "val_score_mean": float(np.mean(val_scores)),
            "val_score_std": float(np.std(val_scores)),
        }
        self._save(index=index, metrics=metrics)
        return metrics

    def _extract_embeddings(self, loader: DataLoader[Tensor]) -> np.ndarray:
        outputs: list[np.ndarray] = []
        with torch.inference_mode():
            for batch in loader:
                batch = batch.to(self.device, non_blocking=True)
                features = self.model(batch)
                outputs.append(features.detach().cpu().numpy().astype(np.float32))
        return np.concatenate(outputs, axis=0)

    def _build_index(self, embeddings: np.ndarray) -> faiss.IndexFlatL2:
        index = faiss.IndexFlatL2(embeddings.shape[1])
        index.add(embeddings)
        return index

    @staticmethod
    def _knn_score(index: faiss.IndexFlatL2, embeddings: np.ndarray, k: int) -> np.ndarray:
        distances, _ = index.search(embeddings, k)
        return np.mean(distances, axis=1)

    def _save(self, index: faiss.IndexFlatL2, metrics: dict[str, float | str | int]) -> None:
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(output_dir / "faiss.index"))
        metadata = {
            "config": asdict(self.config),
            "metrics": metrics,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
