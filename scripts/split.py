import argparse
import json
import random
import shutil
from pathlib import Path


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default="datasets/normal/images")
    parser.add_argument("--train-dir", default="data/train/normal")
    parser.add_argument("--val-dir", default="data/val/normal")
    parser.add_argument("--val-ratio", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--clean", action="store_true")
    return parser.parse_args()


def collect_images(source_dir: Path) -> list[Path]:
    images = [p for p in source_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(images)


def remove_dir_contents(directory: Path) -> None:
    if not directory.exists():
        return
    for child in directory.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def copy_group(files: list[Path], source_root: Path, target_root: Path) -> None:
    for file_path in files:
        relative = file_path.relative_to(source_root)
        destination = target_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file_path, destination)


def main() -> None:
    args = parse_args()
    source_dir = Path(args.source_dir)
    train_dir = Path(args.train_dir)
    val_dir = Path(args.val_dir)
    if not source_dir.exists():
        raise FileNotFoundError(f"source_dir 不存在: {source_dir}")
    if not 0.0 < args.val_ratio < 1.0:
        raise ValueError("val_ratio 必须在 (0, 1) 区间")

    images = collect_images(source_dir)
    if not images:
        raise ValueError(f"source_dir 中没有可用图像: {source_dir}")

    shuffled = images[:]
    random.Random(args.seed).shuffle(shuffled)
    val_count = int(len(shuffled) * args.val_ratio)
    train_files = shuffled[val_count:]
    val_files = shuffled[:val_count]
    if not train_files or not val_files:
        raise ValueError("划分后 train 或 val 为空，请调整 val_ratio 或增加样本")

    train_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)
    if args.clean:
        remove_dir_contents(train_dir)
        remove_dir_contents(val_dir)

    copy_group(train_files, source_root=source_dir, target_root=train_dir)
    copy_group(val_files, source_root=source_dir, target_root=val_dir)

    result = {
        "source_dir": str(source_dir),
        "train_dir": str(train_dir),
        "val_dir": str(val_dir),
        "total_images": len(shuffled),
        "train_images": len(train_files),
        "val_images": len(val_files),
        "val_ratio": args.val_ratio,
        "seed": args.seed,
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
