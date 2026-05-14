import matplotlib.pyplot as plt
import torch
from paths import MODELS_DIR, REPORT_DIR, TEST_DIR
from PIL import Image
from torch import nn
from torchvision import models, transforms

MODEL_PATH = MODELS_DIR / "resnet18_transfer.pt"
CLASSES_PATH = MODELS_DIR / "resnet18_transfer_classes.txt"
CLASS_NAMES_PATH = MODELS_DIR / "resnet18_transfer_classes.txt"
OUTPUT_PATH = REPORT_DIR / "transfer_predictions_gallery.png"
IMAGES_PER_CLASS = 3


def load_class_names():
    if not CLASS_NAMES_PATH.exists():
        print(f"Error: class file not found: {CLASS_NAMES_PATH}")
        print("Train the transfer model first.")
        raise SystemExit(1)
    with open(CLASS_NAMES_PATH, "r") as f:
        class_names = [line.strip() for line in f.readlines() if line.strip()]
    return class_names


def load_model(class_names):
    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        print("Train the transfer model first.")
        raise SystemExit(1)
    model = models.resnet18(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def select_test_images(class_names):
    selected_images = []
    for class_name in class_names:
        class_dir = TEST_DIR / class_name
        image_paths = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        for image_path in image_paths[:IMAGES_PER_CLASS]:
            selected_images.append((image_path, class_name))
    return selected_images


def predict_image(model, class_names, image_path, transform):
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0)
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)
    predicted_class = class_names[predicted_index.item()]
    return predicted_class, confidence.item()


def create_gallery(model, class_names):
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    selected_images = select_test_images(class_names)

    rows = len(class_names)
    columns = IMAGES_PER_CLASS

    figure, axes = plt.subplots(rows, columns, figsize=(10, 8))

    for axis, (image_path, true_class) in zip(
        axes.flatten(), selected_images, strict=False
    ):
        predicted_class, confidence = predict_image(
            model, class_names, image_path, transform
        )
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            axis.imshow(image)
            axis.axis("off")
            axis.set_title(
                f"True: {true_class}\nPred: {predicted_class}\nConf: {confidence:.2f}"
            )
    plt.tight_layout()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(OUTPUT_PATH)
    plt.close()

    print(f"Saved prediction gallery: {OUTPUT_PATH}")


def main():
    class_names = load_class_names()
    model = load_model(class_names)
    create_gallery(model, class_names)


if __name__ == "__main__":
    main()
