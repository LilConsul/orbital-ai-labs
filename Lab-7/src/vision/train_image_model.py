import time

import joblib
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from src.paths import MODEL_PATH, PROCESSED_ROOT
from src.vision.feature_extractor import extract_features

PLOT_PATH = MODEL_PATH.parent / "accuracy_vs_training_time.png"
MODELS_PATH = MODEL_PATH.parent / "image_classifiers.joblib"


def load_image_split(path):
    X = []
    y = []
    class_dirs = sorted([path for path in path.iterdir() if path.is_dir()])
    for class_dir in class_dirs:
        class_name = class_dir.name
        image_files = sorted(
            [
                path
                for path in class_dir.iterdir()
                if path.suffix.lower() in [".jpg", ".jpeg", ".png"]
            ]
        )
        for image_path in image_files:
            with Image.open(image_path) as image:
                features = extract_features(image)

            X.append(features)
            y.append(class_name)

    X = np.array(X)
    y = np.array(y)

    return X, y


def load_training_and_test_data():
    train_dir = PROCESSED_ROOT / "train"
    test_dir = PROCESSED_ROOT / "test"

    X_train, y_train = load_image_split(train_dir)
    X_test, y_test = load_image_split(test_dir)

    print("=== Image ML Dataset ===")
    print(f"X_train shape: {X_train.shape}")
    print(f"y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}")
    print(f"y_test shape: {y_test.shape}")
    return X_train, X_test, y_train, y_test


def build_models():
    return {
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "KNN": KNeighborsClassifier(n_neighbors=3),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "SVM": SVC(),
    }


def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    print("\n=== Training Image Classifiers ===")
    print(f"Training samples: {len(X_train)}")
    print(f"Number of features per image: {X_train.shape[1]}")

    results = []
    trained_models = {}

    for model_name, model in build_models().items():
        print(f"\n--- {model_name} ---")
        start_time = time.time()
        model.fit(X_train, y_train)
        end_time = time.time()
        training_time = end_time - start_time

        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        trained_models[model_name] = model
        results.append(
            {
                "model_name": model_name,
                "accuracy": accuracy,
                "training_time": training_time,
                "model": model,
            }
        )

        print(f"Model: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Training time: {training_time:.2f} s")

    return trained_models, results


def plot_accuracy_vs_training_time(results):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    for result in results:
        plt.scatter(result["training_time"], result["accuracy"], s=120)
        plt.annotate(
            result["model_name"],
            (result["training_time"], result["accuracy"]),
            textcoords="offset points",
            xytext=(8, 6),
        )

    plt.xlabel("Training time (s)")
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy vs Training Time")
    plt.grid(True, linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig(PLOT_PATH)
    plt.close()

    print("\n=== Saving Plot ===")
    print(f"Saved plot: {PLOT_PATH}")


def save_models(trained_models):
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(trained_models, MODELS_PATH)

    default_model = trained_models["Random Forest"]
    joblib.dump(default_model, MODEL_PATH)

    print("\n=== Saving Models ===")
    print(f"Saved model collection: {MODELS_PATH}")
    print(f"Saved default model: {MODEL_PATH}")


def main():
    X_train, X_test, y_train, y_test = load_training_and_test_data()

    trained_models, results = train_and_evaluate_models(
        X_train, X_test, y_train, y_test
    )
    plot_accuracy_vs_training_time(results)
    save_models(trained_models)


if __name__ == "__main__":
    main()
