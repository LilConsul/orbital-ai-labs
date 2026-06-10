import matplotlib.pyplot as plt
import numpy as np
import torch
from paths import (
    IMAGE_DIR,
    MASK_DIR,
    MODEL_DIR,
    PROJECT_DIR,
    REPORT_DIR,
    get_image_path,
    REPORT_DIR_HARD,
)
from PIL import Image
from segmentation.unet_model import SmallUNet
from torchvision import transforms

NUM_CLASSES = 4

PALETTE = {0: (80, 80, 80), 1: (40, 140, 40), 2: (40, 80, 180), 3: (180, 180, 180)}


def mask_to_rgb(mask):
    height, width = mask.shape
    rgb = np.zeros((height, width, 3), dtype=np.uint8)

    for class_id, color in PALETTE.items():
        rgb[mask == class_id] = color
    return rgb


def load_model(model_path):
    if not model_path.exists():
        print(f"Error: model not found: {model_path.relative_to(PROJECT_DIR)}")
        print("Run train_segmentation.py first.")
        raise SystemExit(1)
    model = SmallUNet(num_classes=NUM_CLASSES)
    state_dict = torch.load(model_path, map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()
    return model


def predict_mask(model, image):
    transform = transforms.Compose([transforms.ToTensor()])
    tensor = transform(image).unsqueeze(0)
    with torch.no_grad():
        output = model(tensor)
        prediction = torch.argmax(output, dim=1)

    return prediction[0].numpy()


def visualize(image, ground_truth, prediction, output_path):
    gt_rgb = mask_to_rgb(ground_truth)

    pred_rgb = mask_to_rgb(prediction)

    plt.figure(figsize=(12, 4))

    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Input image")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(gt_rgb)
    plt.title("Ground truth mask")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(pred_rgb)
    plt.title("Predicted mask")
    plt.axis("off")

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    plt.savefig(output_path)
    plt.show()
    print(f"Saved prediction: {output_path.relative_to(PROJECT_DIR)}")


def main():
    MODEL_PATH = MODEL_DIR / "small_unet.pt"
    for i in range(1, 6):
        IMAGE_PATH = get_image_path(IMAGE_DIR, img_num=i)
        MASK_PATH = get_image_path(MASK_DIR, img_num=i)
        OUTPUT_PATH = REPORT_DIR_HARD / f"prediction_{i}.png"

        model = load_model(MODEL_PATH)
        image = Image.open(IMAGE_PATH).convert("RGB")
        ground_truth = np.array(Image.open(MASK_PATH), dtype=np.int64)
        prediction = predict_mask(model, image)
        visualize(image, ground_truth, prediction, OUTPUT_PATH)


if __name__ == "__main__":
    main()
