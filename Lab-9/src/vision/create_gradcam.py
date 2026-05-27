import matplotlib.pyplot as plt
import numpy as np
import torch
from gradcam_model import GradCAM
from paths import MODELS_DIR, REPORT_DIR, TEST_DIR
from PIL import Image
from torch import nn
from torchvision import models, transforms

MODEL_PATH = MODELS_DIR / "resnet18_transfer.pt"
CLASS_NAMES_PATH = MODELS_DIR / "resnet18_transfer.txt"
OUTPUT_PATH = REPORT_DIR / "gradcam_example.png"


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


def load_image(image_path):
    if not image_path.exists():
        print(f"Error: image not found: {image_path}")
        raise SystemExit(1)
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    display_transform = transforms.Compose([transforms.Resize((224, 224))])
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        display_image = display_transform(image)
        image_tensor = transform(image)

    image_tensor = image_tensor.unsqueeze(0)

    return image_tensor, display_image


def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)
        confidence, predicted_index = torch.max(probabilities, dim=1)

    predicted_class = class_names[predicted_index.item()]
    return predicted_index.item(), predicted_class, confidence.item()


def save_gradcam_visualization(display_image, cam, predicted_class, confidence):
    image_array = np.array(display_image)
    cam_image = Image.fromarray(np.uint8(cam * 255))
    cam_image = cam_image.resize(display_image.size)
    cam_resized = np.array(cam_image) / 255.0
    figure, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image_array)
    axes[0].set_title("Original image")
    axes[0].axis("off")
    axes[1].imshow(cam_resized, cmap="jet")
    axes[1].set_title("Grad-CAM heatmap")
    axes[1].axis("off")
    axes[2].imshow(image_array)
    axes[2].imshow(cam_resized, cmap="jet", alpha=0.45)
    axes[2].set_title(f"Prediction: {predicted_class}\nConfidence: {confidence:.2f}")
    axes[2].axis("off")
    plt.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUTPUT_PATH)
    plt.close()
    print(f"Saved Grad-CAM visualization: {OUTPUT_PATH}")


def main():
    def get_example_image(num: int = 0):
        class_names = load_class_names()
        return next(TEST_DIR.glob(f"{class_names[num % len(class_names)]}/*"))

    for IMAGE_PATH in [get_example_image(x) for x in range(3)]:
        class_names = load_class_names()
        model = load_model(class_names)
        image_tensor, display_image = load_image(IMAGE_PATH)
        predicted_index, predicted_class, confidence = predict(
            model, image_tensor, class_names
        )
        print("=== Grad-CAM Explanation ===")
        print(f"Image: {IMAGE_PATH}")
        print(f"Predicted class: {predicted_class}")
        print(f"Confidence: {confidence:.4f}")
        gradcam = GradCAM(model=model, target_layer=model.layer4)
        cam = gradcam.generate(
            image_tensor=image_tensor, target_class_index=predicted_index
        )
        gradcam.close()
        save_gradcam_visualization(
            display_image=display_image,
            cam=cam,
            predicted_class=predicted_class,
            confidence=confidence,
        )


if __name__ == "__main__":
    main()
