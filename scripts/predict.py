import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
import torch
from PIL import Image
from torchvision import transforms

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from one_class.model import FeatureExtractor, resolve_device


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    threshold = float(metadata["metrics"]["threshold"])
    device = resolve_device(args.device or config["device"])
    model = FeatureExtractor(backbone=config["backbone"]).to(device).eval()
    tfm = transforms.Compose(
        [
            transforms.Resize((config["image_size"], config["image_size"])),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )
    image = Image.open(args.image).convert("RGB")
    tensor = tfm(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        embedding = model(tensor).detach().cpu().numpy().astype(np.float32)
    index = faiss.read_index(str(artifact_dir / "faiss.index"))
    distances, _ = index.search(embedding, int(config["knn_k"]))
    score = float(np.mean(distances, axis=1)[0])
    prediction = "normal" if score <= threshold else "anomaly"
    print(
        json.dumps(
            {
                "prediction": prediction,
                "score": score,
                "threshold": threshold,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
