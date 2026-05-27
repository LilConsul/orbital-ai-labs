import csv
import math
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from paths import (
    GRADCAM_EXAMPLES_DIR,
    MODELS_TRANSFER_CLASS_PATH,
    MODELS_TRANSFER_MODEL_PATH,
    PROJECT_DIR,
    TEST_DIR,
    get_image_path,
)
from PIL import Image, ImageEnhance, ImageFilter
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


def pil_to_model_tensor(pil_image):
    """Convert a PIL image to the model input tensor (batch dim added)."""
    transform = transforms.Compose(
        [transforms.Resize((224, 224)), transforms.ToTensor()]
    )
    return transform(pil_image).unsqueeze(0)


def get_transformed_versions(pil_image):
    versions = {}
    versions["original"] = pil_image.copy()
    versions["hflip"] = pil_image.transpose(Image.FLIP_LEFT_RIGHT)
    versions["rotate90"] = pil_image.rotate(90, expand=True)
    versions["gaussian_blur"] = pil_image.filter(ImageFilter.GaussianBlur(radius=2))
    enhancer = ImageEnhance.Brightness(pil_image)
    versions["bright_up"] = enhancer.enhance(1.5)
    versions["bright_down"] = enhancer.enhance(0.6)

    # Add Gaussian noise to the image (keeps same mode)
    arr = np.asarray(pil_image).astype(np.float32) / 255.0
    noise = np.random.normal(loc=0.0, scale=0.05, size=arr.shape).astype(np.float32)
    noisy = np.clip(arr + noise, 0.0, 1.0)
    noisy_img = Image.fromarray((noisy * 255).astype(np.uint8))
    versions["noisy"] = noisy_img

    return versions


def heatmap_center_of_mass(heatmap):
    h = heatmap.shape[0]
    w = heatmap.shape[1]
    cam = heatmap.copy().astype(np.float32)
    total = cam.sum()
    if total <= 0:
        return (w / 2.0, h / 2.0)
    cam_norm = cam / total
    xs = np.arange(w)
    ys = np.arange(h)
    cx = (cam_norm.sum(axis=0) * xs).sum()
    cy = (cam_norm.sum(axis=1) * ys).sum()
    return (cx, cy)


def run_transform_sensitivity(
    image_path, model, class_names, out_dir, max_transforms=None
):
    pil_image = Image.open(image_path).convert("RGB")
    versions = get_transformed_versions(pil_image)
    if max_transforms is not None:
        # keep deterministic order
        keys = list(versions.keys())[:max_transforms]
        versions = {k: versions[k] for k in keys}

    results = {}

    # Process original first
    orig_pil = versions["original"]
    orig_tensor = pil_to_model_tensor(orig_pil)
    orig_pred, orig_conf = predict(model, orig_tensor, class_names)
    orig_heat = create_heatmap(model, orig_tensor, GradCAM)
    orig_center = heatmap_center_of_mass(orig_heat)
    results["original"] = {
        "pred": orig_pred,
        "conf": orig_conf,
        "heat": orig_heat,
        "center": orig_center,
    }

    # Iterate transforms
    for name, img in versions.items():
        if name == "original":
            continue
        tensor = pil_to_model_tensor(img)
        pred, conf = predict(model, tensor, class_names)
        heat = create_heatmap(model, tensor, GradCAM)
        center = heatmap_center_of_mass(heat)
        # compute metrics
        conf_delta = conf - orig_conf
        # Euclidean distance in pixels (heatmap has same spatial dims as model output, usually 7x7 or similar)
        # But create_heatmap returns cam resized to input (224x224) by show_cam_on_image step; here it's native cam dims
        # We'll compute distance in heatmap pixel coordinates and normalize by diagonal
        dx = center[0] - orig_center[0]
        dy = center[1] - orig_center[1]
        dist = math.hypot(dx, dy)
        results[name] = {
            "pred": pred,
            "conf": conf,
            "heat": heat,
            "center": center,
            "conf_delta": conf_delta,
            "center_shift": dist,
        }

    # Create and save comparison figure
    n_versions = len(versions)
    cols = 3
    rows = math.ceil((n_versions + 1) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = np.atleast_1d(axes).ravel()

    # Show original image and overlay
    image_array = np.asarray(orig_pil.resize((224, 224)), dtype=np.float32) / 255.0
    orig_vis = show_cam_on_image(image_array, orig_heat, use_rgb=True)
    axes[0].imshow(orig_vis)
    axes[0].set_title(
        f"original\n{results['original']['pred']} ({results['original']['conf']:.2f})"
    )
    axes[0].axis("off")

    idx = 1
    for name, img in versions.items():
        if name == "original":
            continue
        # prepare overlay from saved heatmap
        img_resized = img.resize((224, 224))
        arr = np.asarray(img_resized, dtype=np.float32) / 255.0
        vis = show_cam_on_image(arr, results[name]["heat"], use_rgb=True)
        title = f"{name}\n{results[name]['pred']} ({results[name]['conf']:.2f})\nΔconf={results[name]['conf_delta']:.3f}\nshift={results[name]['center_shift']:.2f}"
        axes[idx].imshow(vis)
        axes[idx].set_title(title)
        axes[idx].axis("off")
        idx += 1

    for ax in axes[idx:]:
        ax.axis("off")

    fig.suptitle(f"Sensitivity: {image_path.stem}", fontsize=16)
    plt.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = out_dir / f"sensitivity_{image_path.stem}.png"
    plt.savefig(out_path)
    plt.show()

    # Save numeric summary
    summary_lines = [
        f"Image: {image_path.relative_to(PROJECT_DIR)}",
        f"Original: {orig_pred} ({orig_conf:.4f})",
    ]
    for name, res in results.items():
        if name == "original":
            continue
        summary_lines.append(
            f"{name}: pred={res['pred']}, conf={res['conf']:.4f}, Δconf={res.get('conf_delta', 0):.4f}, shift={res.get('center_shift', 0):.4f}"
        )
    summary_text = "\n".join(summary_lines)
    (out_dir / f"sensitivity_{image_path.stem}.txt").write_text(summary_text)

    print(summary_text)
    return results


def compute_iou_from_heatmaps(h1, h2, thresh_quantile=0.7):
    h1n = (h1 - h1.min()) / (h1.max() - h1.min() + 1e-8)
    h2n = (h2 - h2.min()) / (h2.max() - h2.min() + 1e-8)
    thr1 = np.quantile(h1n, thresh_quantile)
    thr2 = np.quantile(h2n, thresh_quantile)
    m1 = h1n >= thr1
    m2 = h2n >= thr2
    inter = np.logical_and(m1, m2).sum()
    union = np.logical_or(m1, m2).sum()
    if union == 0:
        return 0.0
    return inter / union


def center_crop(pil_img, fraction=0.8):
    w, h = pil_img.size
    new_w = int(w * fraction)
    new_h = int(h * fraction)
    left = (w - new_w) // 2
    top = (h - new_h) // 2
    return pil_img.crop((left, top, left + new_w, top + new_h)).resize((w, h))


def random_crop(pil_img, fraction=0.8):
    w, h = pil_img.size
    new_w = int(w * fraction)
    new_h = int(h * fraction)
    left = random.randint(0, w - new_w)
    top = random.randint(0, h - new_h)
    return pil_img.crop((left, top, left + new_w, top + new_h)).resize((w, h))


def run_independent_experiment(model, class_names, out_dir, images_per_class=5):
    grouped = get_sample_images_by_class()
    rows = []

    noise_levels = [0.02, 0.05, 0.1]
    crops = {
        "center_crop": lambda im: center_crop(im, 0.8),
        "random_crop": lambda im: random_crop(im, 0.8),
    }

    def process_transform(
        image_rel,
        pil_img,
        transform_name,
        orig_conf,
        orig_heat,
        orig_center,
        class_name,
    ):
        t = pil_to_model_tensor(pil_img)
        pred, conf = predict(model, t, class_names)
        heat = create_heatmap(model, t, GradCAM)
        center = heatmap_center_of_mass(heat)
        dist = math.hypot(center[0] - orig_center[0], center[1] - orig_center[1])
        iou = compute_iou_from_heatmaps(orig_heat, heat)
        rows.append(
            {
                "image": image_rel,
                "class": class_name,
                "transform": transform_name,
                "pred": pred,
                "conf": conf,
                "conf_delta": conf - orig_conf,
                "center_shift": dist,
                "cam_iou": iou,
            }
        )

    for class_name, paths in grouped.items():
        selected = list(paths)[:images_per_class]
        for image_path in selected:
            image_rel = str(image_path.relative_to(PROJECT_DIR))
            pil = Image.open(image_path).convert("RGB")

            # baseline
            orig_tensor = pil_to_model_tensor(pil)
            orig_pred, orig_conf = predict(model, orig_tensor, class_names)
            orig_heat = create_heatmap(model, orig_tensor, GradCAM)
            orig_center = heatmap_center_of_mass(orig_heat)

            # record original row
            rows.append(
                {
                    "image": image_rel,
                    "class": class_name,
                    "transform": "original",
                    "pred": orig_pred,
                    "conf": orig_conf,
                    "conf_delta": 0.0,
                    "center_shift": 0.0,
                    "cam_iou": 1.0,
                }
            )

            # brightness
            process_transform(
                image_rel,
                ImageEnhance.Brightness(pil).enhance(1.5),
                "bright_up",
                orig_conf,
                orig_heat,
                orig_center,
                class_name,
            )
            process_transform(
                image_rel,
                ImageEnhance.Brightness(pil).enhance(0.6),
                "bright_down",
                orig_conf,
                orig_heat,
                orig_center,
                class_name,
            )

            # flip
            process_transform(
                image_rel,
                pil.transpose(Image.FLIP_LEFT_RIGHT),
                "hflip",
                orig_conf,
                orig_heat,
                orig_center,
                class_name,
            )

            # rotate
            process_transform(
                image_rel,
                pil.rotate(90, expand=True),
                "rotate90",
                orig_conf,
                orig_heat,
                orig_center,
                class_name,
            )

            # gaussian blur
            process_transform(
                image_rel,
                pil.filter(ImageFilter.GaussianBlur(radius=2)),
                "gaussian_blur",
                orig_conf,
                orig_heat,
                orig_center,
                class_name,
            )

            # noise levels
            arr = np.asarray(pil).astype(np.float32) / 255.0
            for nl in noise_levels:
                noise = np.random.normal(0, nl, size=arr.shape).astype(np.float32)
                noisy = np.clip(arr + noise, 0.0, 1.0)
                noisy_img = Image.fromarray((noisy * 255).astype(np.uint8))
                process_transform(
                    image_rel,
                    noisy_img,
                    f"noise_{nl}",
                    orig_conf,
                    orig_heat,
                    orig_center,
                    class_name,
                )

            # crops
            for cname, func in crops.items():
                cimg = func(pil)
                process_transform(
                    image_rel,
                    cimg,
                    cname,
                    orig_conf,
                    orig_heat,
                    orig_center,
                    class_name,
                )

    # write CSV
    out_csv = out_dir / "experiment_summary.csv"
    keys = [
        "image",
        "class",
        "transform",
        "pred",
        "conf",
        "conf_delta",
        "center_shift",
        "cam_iou",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for r in rows:
            writer.writerow(r)

    import pandas as pd

    df = pd.DataFrame(rows)
    # boxplot conf_delta
    plt.figure(figsize=(10, 6))
    df.boxplot(column="conf_delta", by="transform", rot=45)
    plt.title("ΔConfidence by transform")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(out_dir / "box_conf_delta.png")

    plt.figure(figsize=(10, 6))
    df.boxplot(column="center_shift", by="transform", rot=45)
    plt.title("Center shift by transform")
    plt.suptitle("")
    plt.tight_layout()
    plt.savefig(out_dir / "box_center_shift.png")

    print(f"Experiment saved to {out_csv} and summary plots saved.")


def main():
    GRADCAM_EXAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    class_names = load_class_names()
    model = load_model(class_names)

    print("Running Independent Task 1...")
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

    # Independent Task 2: sensitivity to image transformations
    # For each class, run analysis on the first image found
    print("Running Independent Task 2...")
    for class_name, image_paths in class_to_image_paths.items():
        if not image_paths:
            continue
        image_path = image_paths[0]
        print(f"Running transform sensitivity for {class_name} / {image_path.name}")
        run_transform_sensitivity(image_path, model, class_names, GRADCAM_EXAMPLES_DIR)

    # Independent Task 3: run the aggregated experiment across several images per class
    print("Running Independent Task 3...")
    run_independent_experiment(
        model, class_names, GRADCAM_EXAMPLES_DIR, images_per_class=3
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
