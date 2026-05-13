import time

import matplotlib.pyplot as plt
import torch
from paths import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODELS_DIR,
    REPORT_DIR,
    TEST_DIR,
    TRAIN_DIR,
)
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix
from src.vision.image_dataset import EuroSATDataset
from torch import nn
from torch.utils.data import DataLoader
from torchvision import models, transforms

MODEL_PATH = MODELS_DIR / "resnet18_finetuned.pt"
CLASS_NAMES_PATH = MODELS_DIR / "resnet18_finetuned_classes.txt"
REPORT_PATH = REPORT_DIR / "fine_tuning_report.txt"
CONFUSION_MATRIX_PATH = REPORT_DIR / "fine_tuning_confusion_matrix.png"


def create_dataloaders():
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    train_dataset = EuroSATDataset(root_dir=TRAIN_DIR, transform=transform)
    test_dataset = EuroSATDataset(root_dir=TEST_DIR, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    print("=== Transfer Learning DataLoader ===")
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


def create_transfer_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    for parameter in model.parameters():
        parameter.requires_grad = False

    for parameter in model.layer4.parameters():
        parameter.requires_grad = True

    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, num_classes)
    return model


def print_trainable_parameters(model):
    print("\n=== Trainable Parameters ===")
    trainable_count = 0
    total_count = 0
    for name, parameter in model.named_parameters():
        total_count += parameter.numel()
        if parameter.requires_grad:
            trainable_count += parameter.numel()
        print(name)

    print(f"Trainable parameters: {trainable_count}")
    print(f"Total parameters: {total_count}")


def train_model(model, train_loader, device):
    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda parameter: parameter.requires_grad, model.parameters()),
        lr=LEARNING_RATE,
    )
    model.train()
    start_time = time.time()
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
    end_time = time.time()
    training_time = end_time - start_time

    print(f"Training time: {training_time:.2f} seconds")
    return training_time


def evaluate_model(model, test_loader, device, class_names):
    model.eval()
    all_labels = []
    all_predictions = []
    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, dim=1)
            all_labels.extend(labels.numpy())
            all_predictions.extend(predicted.cpu().numpy())
    accuracy = accuracy_score(all_labels, all_predictions)
    print("\n=== Transfer Learning Evaluation ===")
    print(f"Test samples: {len(all_labels)}")
    print(f"Accuracy: {accuracy:.4f}")
    matrix = confusion_matrix(all_labels, all_predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=class_names
    )
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    display.plot()
    plt.title("Transfer Learning Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")
    return accuracy, all_labels, all_predictions


def save_model(model, class_names):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), MODEL_PATH)
    with open(CLASS_NAMES_PATH, "w") as f:
        for class_name in class_names:
            f.write(class_name + "\n")
    print("\n=== Saving Transfer Model ===")
    print(f"Saved model: {MODEL_PATH}")
    print(f"Saved classes: {CLASS_NAMES_PATH}")


def save_report(accuracy, training_time, class_names):
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w") as f:
        f.write("TRANSFER LEARNING REPORT\n")
        f.write("========================\n\n")

        f.write("Model: ResNet18 pretrained on ImageNet\n")
        f.write("Training mode: fine-tuned last ResNet block\n")
        f.write("Classifier: layer4 and final classifier\n\n")

        f.write(f"Classes: {class_names}\n")
        f.write(f"Epochs: {EPOCHS}\n")
        f.write(f"Learning rate: {LEARNING_RATE}\n")
        f.write(f"Training time: {training_time:.2f} seconds\n")
        f.write(f"Test accuracy: {accuracy:.4f}\n\n")

        f.write("Interpretation:\n")
        f.write(
            "The model reused pretrained visual features and trained only "
            "a new classifier head for the EuroSAT subset.\n"
        )
    print(f"Saved report: {REPORT_PATH}")


def save_confusion_matrix(all_labels, all_predictions, class_names):
    matrix = confusion_matrix(all_labels, all_predictions)
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix, display_labels=class_names
    )
    CONFUSION_MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    display.plot()
    plt.title("Transfer Learning Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_PATH)
    plt.close()
    print(f"Saved confusion matrix: {CONFUSION_MATRIX_PATH}")


def main():
    train_loader, test_loader, class_names = create_dataloaders()
    device = get_device()
    print(f"Using device: {device}")
    model = create_transfer_model(num_classes=len(class_names))
    print_trainable_parameters(model)
    model = model.to(device)
    training_time = train_model(model, train_loader, device)
    accuracy, all_labels, all_predictions = evaluate_model(
        model, test_loader, device, class_names
    )
    save_model(model, class_names)
    save_report(accuracy, training_time, class_names)
    save_confusion_matrix(all_labels, all_predictions, class_names)


if __name__ == "__main__":
    main()
