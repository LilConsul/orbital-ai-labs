import math

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from paths import (
    GRADCAM_EXAMPLES_DIR,
    MODELS_TRANSFER_CLASS_PATH,
    MODELS_TRANSFER_MODEL_PATH,
    TEST_DIR,
    get_image_path,
)
from PIL import Image
from pytorch_grad_cam import EigenCAM, GradCAM, HiResCAM, LayerCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import models, transforms

CLASS_NAMES_PATH = MODELS_TRANSFER_CLASS_PATH
MODEL_PATH = MODELS_TRANSFER_MODEL_PATH
CAM_METHODS = (
    ("Grad-CAM", GradCAM),
    ("HiResCAM", HiResCAM),
    ("EigenCAM", EigenCAM),
    ("LayerCAM", LayerCAM),
)


def load_class_names():
    with open(CLASS_NAMES_PATH) as f:
        class_names = [line.strip() for line in f]
    return class_names


def load_model(class_names):
    model = models.resnet18(weights=None)
    input_features = model.fc.in_features
    model.fc = nn.Linear(input_features, len(class_names))
    state_dict = torch.load(MODEL_PATH, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def load_image(image_path):
    image = Image.open(image_path).convert("RGB")

    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    tensor = transform(image)
    return image, tensor.unsqueeze(0)


def predict(model, image_tensor, class_names):
    with torch.no_grad():
        outputs = model(image_tensor)

        probabilities = torch.softmax(outputs, dim=1)

        confidence, predicted = torch.max(probabilities, dim=1)
    predicted_idx = predicted.item()
    predicted_class = class_names[predicted_idx]
    confidence_value = confidence.item()
    print(f"Prediction: {predicted_class}")
    print(f"Confidence: {confidence_value:.4f}\n")
    return predicted_class, confidence_value


def create_heatmap(model, image_tensor, cam_class):
    target_layers = [model.layer4[-1]]
    cam = cam_class(model=model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=image_tensor)
    return grayscale_cam[0]


def render_cam_comparison(image, heatmap, path, title, prediction, confidence):
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32) / np.float32(255.0)

    visualization = show_cam_on_image(image_array, heatmap, use_rgb=True)

    plt.figure(figsize=(10, 5))

    # Original image
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title(f"{title}\nPred: {prediction} ({confidence:.2f})")
    plt.axis("off")

    # Grad-CAM image
    plt.subplot(1, 2, 2)
    plt.imshow(visualization)
    plt.title(f"Grad-CAM\nPred: {prediction} ({confidence:.2f})")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def render_multi_cam_comparison(
    image, image_tensor, model, path, prediction, confidence
):
    image = image.resize((224, 224))
    image_array = np.asarray(image, dtype=np.float32) / np.float32(255.0)

    heatmaps = [
        create_heatmap(model, image_tensor, cam_class) for _, cam_class in CAM_METHODS
    ]

    plt.figure(figsize=(16, 8))
    plt.subplot(2, 3, 1)
    plt.imshow(image)
    plt.title(f"Original\nPred: {prediction} ({confidence:.2f})")
    plt.axis("off")

    for index, ((method_name, _), heatmap) in enumerate(
        zip(CAM_METHODS, heatmaps, strict=True), start=2
    ):
        plt.subplot(2, 3, index)
        plt.imshow(show_cam_on_image(image_array, heatmap, use_rgb=True))
        plt.title(method_name)
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(path)
    plt.show()


def render_class_cam_grid(class_name, image_paths, model, class_names, path, columns=3):
    total_images = len(image_paths)
    columns = min(columns, total_images)
    rows = math.ceil(total_images / columns)

    fig, axes = plt.subplots(rows, columns, figsize=(5 * columns, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    for idx, image_path in enumerate(image_paths):
        image, tensor = load_image(image_path)
        prediction, confidence = predict(model, tensor, class_names)
        heatmap = create_heatmap(model, tensor, GradCAM)

        image = image.resize((224, 224))
        image_array = np.asarray(image, dtype=np.float32) / np.float32(255.0)
        overlay = show_cam_on_image(image_array, heatmap, use_rgb=True)

        axes[idx].imshow(overlay)
        axes[idx].set_title(f"{image_path.stem}\nPred: {prediction} ({confidence:.2f})")
        axes[idx].axis("off")

    for axis in axes[total_images:]:
        axis.axis("off")

    fig.suptitle(f"{class_name}: Grad-CAM", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    plt.savefig(path)
    plt.show()


def get_sample_images_by_class():
    class_dirs = sorted([path for path in TEST_DIR.iterdir() if path.is_dir()])
    grouped_images = {}
    for class_dir in class_dirs:
        image_paths = sorted(class_dir.glob("*.jpg"))
        if image_paths:
            grouped_images[class_dir.name] = image_paths
    return grouped_images


def main():
    GRADCAM_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names()
    model = load_model(class_names)

    class_to_image_paths = get_sample_images_by_class()
    for class_name, image_paths in class_to_image_paths.items():
        safe_class_name = class_name.lower().replace(" ", "_")
        render_class_cam_grid(
            class_name,
            image_paths,
            model,
            class_names,
            GRADCAM_EXAMPLES_DIR / f"gradcam_{safe_class_name}.png",
        )

    comparison_image_path = get_image_path(TEST_DIR, f_num=0, img_num=0)
    comparison_image, comparison_tensor = load_image(comparison_image_path)
    comparison_prediction, comparison_confidence = predict(
        model, comparison_tensor, class_names
    )
    render_multi_cam_comparison(
        comparison_image,
        comparison_tensor,
        model,
        GRADCAM_EXAMPLES_DIR / f"cam_compare_{comparison_image_path.stem}.png",
        comparison_prediction,
        comparison_confidence,
    )


if __name__ == "__main__":
    main()
