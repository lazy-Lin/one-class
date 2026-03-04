import argparse
import json
from pathlib import Path

import faiss
import numpy as np
import onnxruntime as ort
from PIL import Image
from torchvision import transforms


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-dir", required=True)
    parser.add_argument("--onnx", required=True)
    parser.add_argument("--image", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = Path(args.artifact_dir)
    metadata = json.loads((artifact_dir / "metadata.json").read_text(encoding="utf-8"))
    config = metadata["config"]
    threshold = float(metadata["metrics"]["threshold"])
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
    tensor = tfm(image).unsqueeze(0).numpy().astype(np.float32)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    embedding = session.run(["embedding"], {"input": tensor})[0].astype(np.float32)
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
