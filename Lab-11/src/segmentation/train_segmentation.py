import random

import torch
from paths import IMAGE_DIR, MASK_DIR, MODEL_DIR, PROJECT_DIR, REPORT_DIR_CLASSES
from segmentation.segmentation_dataset import SyntheticSegmentationDataset
from segmentation.unet_model import SmallUNet
from torch import nn
from torch.utils.data import DataLoader, random_split
from torchvision import transforms

MODEL_PATH = MODEL_DIR / "small_unet.pt"
REPORT_PATH = REPORT_DIR_CLASSES / "segmentation_report.txt"

BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 4
RANDOM_SEED = 42


def create_dataloaders():
    transform = transforms.Compose([transforms.ToTensor()])
    dataset = SyntheticSegmentationDataset(
        image_dir=IMAGE_DIR, mask_dir=MASK_DIR, transform=transform
    )
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(RANDOM_SEED),
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    print("=== Segmentation DataLoaders ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")

    images, masks = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch mask shape: {masks.shape}")

    return train_loader, test_loader


def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(model, train_loader, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_function(outputs, masks)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        average_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_loss:.4f}")


def evaluate_model(model, test_loader, device):
    model.eval()
    correct_pixels = 0
    total_pixels = 0

    class_correct = torch.zeros(NUM_CLASSES, dtype=torch.long)
    class_total = torch.zeros(NUM_CLASSES, dtype=torch.long)

    with torch.no_grad():
        for images, masks in test_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            predictions = torch.argmax(outputs, dim=1)

            # Overall accuracy
            correct_pixels += (predictions == masks).sum().item()
            total_pixels += masks.numel()

            # Per-class accuracy
            for class_id in range(NUM_CLASSES):
                class_mask = masks == class_id
                class_correct[class_id] += (
                    ((predictions == class_id) & class_mask).sum().item()
                )
                class_total[class_id] += class_mask.sum().item()

    accuracy = correct_pixels / total_pixels

    class_accuracies = {}
    class_names = {0: "background", 1: "vegetation", 2: "water", 3: "urban"}

    print("\n=== Segmentation Evaluation ===")
    print(f"Overall pixel accuracy: {accuracy:.4f}")
    print("\nClass-wise pixel accuracy:")

    for class_id in range(NUM_CLASSES):
        if class_total[class_id] > 0:
            class_acc = class_correct[class_id].item() / class_total[class_id].item()
            class_accuracies[class_id] = class_acc
            print(
                f"  {class_names[class_id]:12s}: {class_acc:.4f} ({class_correct[class_id]}/{class_total[class_id]} pixels)"
            )
        else:
            class_accuracies[class_id] = 0.0
            print(f"  {class_names[class_id]:12s}: N/A (no pixels in test set)")

    return accuracy, class_accuracies


def save_model(model):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    print(f"Saved model: {MODEL_PATH.relative_to(PROJECT_DIR)}")


def save_report(accuracy, class_accuracies):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    class_names = {0: "background", 1: "vegetation", 2: "water", 3: "urban"}

    with open(REPORT_PATH, "w") as f:
        f.write("SEMANTIC SEGMENTATION REPORT\n")
        f.write("============================\n\n")
        f.write("Model: Small U-Net\n")
        f.write("Dataset: synthetic EO segmentation dataset\n")
        f.write(f"Classes: {NUM_CLASSES}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n\n")

        f.write("EVALUATION METRICS\n")
        f.write("------------------\n")
        f.write(f"Overall pixel accuracy: {accuracy:.4f}\n\n")

        f.write("Class-wise pixel accuracy:\n")
        for class_id in range(NUM_CLASSES):
            class_name = class_names[class_id]
            class_acc = class_accuracies[class_id]
            f.write(f"  • {class_name:12s}: {class_acc:.4f}\n")

    print(f"Saved report: {REPORT_PATH.relative_to(PROJECT_DIR)}")


def main():
    random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    train_loader, test_loader = create_dataloaders()
    device = get_device()
    print(f"Using device: {device}")
    model = SmallUNet(num_classes=NUM_CLASSES)
    model = model.to(device)
    train_model(model, train_loader, device)
    accuracy, class_accuracies = evaluate_model(model, test_loader, device)
    save_model(model)
    save_report(accuracy, class_accuracies)


if __name__ == "__main__":
    main()
