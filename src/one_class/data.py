from pathlib import Path
from typing import Iterable

from PIL import Image
from torch import Tensor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


class ImagePathDataset(Dataset[Tensor]):
    def __init__(self, image_paths: list[Path], image_size: int) -> None:
        self.image_paths = image_paths
        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, index: int) -> Tensor:
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        return self.transform(image)


def list_images(directory: Path) -> list[Path]:
    files = [p for p in directory.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(files)


def make_loader(
    image_paths: Iterable[Path],
    image_size: int,
    batch_size: int,
    num_workers: int,
    shuffle: bool,
) -> DataLoader[Tensor]:
    dataset = ImagePathDataset(list(image_paths), image_size=image_size)
    return DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
