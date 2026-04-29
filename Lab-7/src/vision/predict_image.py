from pathlib import Path

import joblib
from matplotlib import pyplot as plt
from PIL import Image
from src.paths import MODEL_PATH, PROCESSED_ROOT
from src.vision.feature_extractor import extract_features

MODELS_PATH = MODEL_PATH.parent / "image_classifiers.joblib"


def load_models():
    if MODELS_PATH.exists():
        models = joblib.load(MODELS_PATH)
        print("Model collection loaded.")
        return models

    if not MODEL_PATH.exists():
        print(f"Error: model file not found: {MODEL_PATH}")
        raise SystemExit(1)

    model = joblib.load(MODEL_PATH)
    print("Single model loaded.")
    return {"Random Forest": model}


def predict_image(models, image_path):
    path = Path(image_path)
    if not path.exists():
        print(f"Error: file not found: {image_path}")
        return

    with Image.open(path) as image:
        features = extract_features(image)
        image_for_plot = image.copy()

    predictions = {}
    for model_name, model in models.items():
        predictions[model_name] = model.predict([features])[0]

    print("\n=== Prediction ===")
    print(f"Image: {image_path}")
    for model_name, prediction in predictions.items():
        print(f"{model_name}: {prediction}")

    title = " | ".join(
        f"{model_name}: {prediction}" for model_name, prediction in predictions.items()
    )

    plt.figure(figsize=(10, 6))
    plt.imshow(image_for_plot)
    plt.title(title)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


def main():
    models = load_models()
    image_path = PROCESSED_ROOT / "test" / "forest" / "forest_0000.jpg"
    predict_image(models, image_path)


if __name__ == "__main__":
    main()
