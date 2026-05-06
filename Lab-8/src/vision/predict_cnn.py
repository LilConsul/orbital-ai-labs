from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from src.paths import (
    CLASS_NAMES_PATH,
    CONFUSION_MATRIX_PATH,
    DATA_DIR,
    MODEL_PATH,
    PROJECT_ROOT,
    REPORT_DIR,
    TEST_DIR,
)
from src.vision.cnn_model import SimpleCNN
from torchvision import transforms


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        print(f"Error: class file not found: {CLASS_NAMES_PATH}")
        print("Train the model first.")
        raise SystemExit(1)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    return class_names


def load_model(class_names):
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        print("Train the model first.")
        raise SystemExit(1)
    model = SimpleCNN(num_classes=len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_image(model, class_names, image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: image not found: {path}")
        return
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    with Image.open(path) as image:
        image = image.convert("RGB")
        image_for_plot = image.copy()
        image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_index.item()]

    print("=== CNN Prediction ===")
    print(f"Image: {image_path}")
    print(f"Predicted class: {predicted_class}")
    print(f"Confidence: {confidence.item():.4f}")

    plt.imshow(image_for_plot)
    plt.title(f"Prediction: {predicted_class}\nConfidence: {confidence.item():.4f}")
    plt.axis("off")
    plt.show()


def generate_confusion_matrix(model, class_names):
    transform = transforms.Compose([transforms.Resize((64, 64)), transforms.ToTensor()])
    y_true = []
    y_pred = []
    for class_name in class_names:
        class_dir = TEST_DIR / class_name
        for image_path in class_dir.iterdir():
            if image_path.suffix.lower() not in [".jpg", ".jpeg", ".png"]:
                continue
            with Image.open(image_path) as image:
                image = image.convert("RGB")
                image_tensor = transform(image)
            image_tensor = image_tensor.unsqueeze(0)
            with torch.no_grad():
                outputs = model(image_tensor)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted_index = torch.max(probabilities, dim=1)
            y_true.append(class_name)
            y_pred.append(class_names[predicted_index.item()])
    cm = confusion_matrix(y_true, y_pred, labels=class_names)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")

    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    with CONFUSION_MATRIX_PATH.open("wb") as f:
        plt.savefig(f, format="png")
        print(
            f"Saved confusion matrix to {CONFUSION_MATRIX_PATH.relative_to(PROJECT_ROOT)}"
        )

    plt.show()


def main():
    class_names = load_class_names()
    model = load_model(class_names)
    image_path = TEST_DIR / "forest" / "forest_0000.jpg"
    predict_image(model, class_names, image_path)

    noise_image_path = DATA_DIR / "inference_samples" / "noise.jpg"
    predict_image(model, class_names, noise_image_path)

    generate_confusion_matrix(model, class_names)


if __name__ == "__main__":
    main()
