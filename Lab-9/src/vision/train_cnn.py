import time

import torch
from paths import MODELS_DIR
from src.paths import (
    BATCH_SIZE,
    CLASS_NAMES_PATH,
    EPOCHS,
    LEARNING_RATE,
    TEST_DIR,
    TRAIN_DIR,
)
from src.vision.cnn_model import SimpleCNN
from src.vision.image_dataset import EuroSATDataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

MODEL_PATH = MODELS_DIR / "cnn_model.pt"


def create_dataloaders():
    transform = transforms.Compose(
        [
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ToTensor(),
        ]
    )
    train_dataset = EuroSATDataset(root_dir=TRAIN_DIR, transform=transform)
    test_dataset = EuroSATDataset(root_dir=TEST_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("=== DataLoader Inspection ===")
    print(f"Training samples: {len(train_dataset)}")
    print(f"Testing samples: {len(test_dataset)}")
    print(f"Classes: {train_dataset.class_names}")
    print(f"Class mapping: {train_dataset.class_to_index}")
    images, labels = next(iter(train_loader))
    print(f"Batch image shape: {images.shape}")
    print(f"Batch label shape: {labels.shape}")
    return train_loader, test_loader, train_dataset.class_names


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def train_model(model, train_loader, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)

            loss = loss_function(outputs, labels)
            loss.backward()

            optimizer.step()

            total_loss += loss.item()

        average_loss = total_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{EPOCHS}, Loss: {average_loss:.4f}")


def evaluate_model(model, test_loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    print("=== Evaluation ===")
    print(f"Test samples: {total}")
    print(f"Accuracy: {accuracy:.4f}")
    return accuracy


def save_model(model, class_names):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        for class_name in class_names:
            f.write(class_name + "\n")
    print("=== Saving Model ===")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved classes: {CLASS_NAMES_PATH}")


def main():
    train_loader, test_loader, class_names = create_dataloaders()
    device = get_device()
    print(f"Using device: {device}")
    model = SimpleCNN(num_classes=len(class_names))
    model = model.to(device)
    time_start = time.time()
    train_model(model, train_loader, device)
    time_end = time.time()
    print(f"Training time: {time_end - time_start}")
    evaluate_model(model, test_loader, device)
    save_model(model, class_names)


if __name__ == "__main__":
    main()
