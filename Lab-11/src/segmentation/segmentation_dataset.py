from pathlib import Path

import numpy as np
import torch
from paths import IMAGE_DIR, MASK_DIR
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


class SyntheticSegmentationDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.image_paths = sorted(self.image_dir.glob("*.png"))
        if len(self.image_paths) == 0:
            print("Error: no images found.")
            print("Run generate_synthetic_dataset.py first.")
            raise SystemExit(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        mask_path = self.mask_dir / image_path.name
        image = Image.open(str(image_path)).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = transforms.ToTensor()(image)

        mask_array = np.array(mask, dtype=np.int64)
        mask_tensor = torch.from_numpy(mask_array)
        return image, mask_tensor


def main():
    dataset = SyntheticSegmentationDataset(image_dir=IMAGE_DIR, mask_dir=MASK_DIR)
    image, mask = dataset[0]
    print("=== Segmentation Dataset Inspection ===")
    print(f"Number of samples: {len(dataset)}")
    print(f"Image shape: {image.shape}")
    print(f"Mask shape: {mask.shape}")
    print(f"Mask classes: {torch.unique(mask)}")


if __name__ == "__main__":
    main()
