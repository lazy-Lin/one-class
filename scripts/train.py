import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from one_class.pipeline import OneClassFaissPipeline, PipelineConfig
from config import CONFIG


def main() -> None:
    config = PipelineConfig(
        train_dir=CONFIG.train_dir,
        val_dir=CONFIG.val_dir,
        output_dir=CONFIG.output_dir,
        backbone=CONFIG.backbone,
        image_size=CONFIG.image_size,
        batch_size=CONFIG.batch_size,
        num_workers=CONFIG.num_workers,
        knn_k=CONFIG.knn_k,
        threshold_quantile=CONFIG.threshold_quantile,
        device=CONFIG.device,
        pretrained_weights=CONFIG.pretrained_weights,
    )
    metrics = OneClassFaissPipeline(config).run()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
