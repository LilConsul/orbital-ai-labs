from src.paths import RAW_DIR
from torchvision.datasets import EuroSAT


def main():
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    dataset = EuroSAT(root=RAW_DIR, download=True)

    print(f"Downloading EuroSAT dataset with {len(dataset)} samples")
    print(f"Classes: {dataset.classes}")


if __name__ == "__main__":
    main()
