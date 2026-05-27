import matplotlib.pyplot as plt
import numpy as np
import torch
from create_gradcam import (
    load_class_names,
    load_image,
    load_model,
    predict,
)
from paths import GRADCAM_EXAMPLES_DIR, TEST_DIR, get_image_path


def create_occlusion_map(
    model, image_tensor, predicted_class, patch_size=32, stride=16
):
    _, _, H, W = image_tensor.shape
    sensitivity = np.zeros((H, W))
    with torch.no_grad():
        original_output = model(image_tensor)
        original_probability = torch.softmax(original_output, dim=1)[
            0, predicted_class
        ].item()

    for y in range(0, H - patch_size, stride):
        for x in range(0, W - patch_size, stride):
            occluded = image_tensor.clone()
            occluded[:, :, y : y + patch_size, x : x + patch_size] = 0
            with torch.no_grad():
                output = model(occluded)
                probability = torch.softmax(output, dim=1)[0, predicted_class].item()
            drop = original_probability - probability
            sensitivity[y : y + patch_size, x : x + patch_size] += drop
    sensitivity -= sensitivity.min()
    sensitivity /= sensitivity.max() + 1e-8
    return sensitivity


def visualize_occlusion(image, sensitivity):
    image = image.resize((224, 224))
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")
    plt.subplot(1, 2, 2)
    plt.imshow(sensitivity, cmap="jet")
    plt.title("Occlusion Sensitivity")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(GRADCAM_EXAMPLES_DIR / "occlusion_example.png")
    plt.show()


def main():
    class_names = load_class_names()
    model = load_model(class_names)
    image_path = get_image_path(TEST_DIR, f_num=0, img_num=0)
    image, tensor = load_image(image_path)
    predicted_class_name, _ = predict(model, tensor, class_names)
    predicted_class_idx = class_names.index(predicted_class_name)
    sensitivity = create_occlusion_map(model, tensor, predicted_class_idx)
    visualize_occlusion(image, sensitivity)


if __name__ == "__main__":
    main()
