from pathlib import Path

from PIL import Image
from src.paths import TRAIN_DIR
from torch.utils.data import Dataset
from torchvision import transforms


class EuroSATDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = Path(root_dir)
        self.transform = transform
        if not self.root_dir.exists():
            print(f"Error: dataset directory does not exist: {self.root_dir}")
            print("Run src/data/download_eurosat.py first.")
            print("Then run src/data/prepare_image_dataset.py.")
            raise SystemExit(1)
        self.samples = []
        self.class_names = sorted(
            [path.name for path in self.root_dir.iterdir() if path.is_dir()]
        )
        self.class_to_index = {
            class_name: index for index, class_name in enumerate(self.class_names)
        }
        for class_name in self.class_names:
            class_dir = self.root_dir / class_name
            image_files = sorted(
                [
                    path
                    for path in class_dir.iterdir()
                    if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
                ]
            )
            for image_path in image_files:
                label = self.class_to_index[class_name]
                self.samples.append((image_path, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        with Image.open(image_path) as image:
            image = image.convert("RGB")

            if self.transform is not None:
                image = self.transform(image)
                return image, label


def main():
    transform = transforms.ToTensor()
    dataset = EuroSATDataset(root_dir=TRAIN_DIR, transform=transform)
    print("=== PyTorch Dataset Inspection ===")
    print(f"Classes: {dataset.class_names}")
    print(f"Class mapping: {dataset.class_to_index}")
    print(f"Number of samples: {len(dataset)}")
    image_tensor, label = dataset[0]
    print(f"First image tensor shape: {image_tensor.shape}")
    print(f"First label: {label}")


if __name__ == "__main__":
    main()
