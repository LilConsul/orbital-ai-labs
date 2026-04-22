from pathlib import Path
from pprint import pformat
from typing import cast

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODELS_FEATURES_PATH = PROCESSED_DIR / "model_features.csv"
MODELS_LABELS_PATH = PROCESSED_DIR / "model_labels.csv"
MODEL_PLAYGROUND_SUMMARY_PATH = REPORTS_DIR / "model_playground_summary.txt"
MODEL_COMPARISON_PANEL_PATH = REPORTS_DIR / "model_comparison_panel.png"
ANOMALY_FLAG_COLUMN = "anomaly_flag"

PATHS = [
    PROCESSED_DIR,
    MODELS_FEATURES_PATH,
    MODELS_LABELS_PATH
]


def validate_input_files():
    print("=== Validating Input Files ===")
    invalid_files = []
    for path in PATHS:
        if not path.exists():
            invalid_files.append(path.relative_to(PROJECT_ROOT))

    if len(invalid_files) > 0:
        print("Error: missing required input file(s):")
        for file in invalid_files:
            print(f"\t- {file}")
        raise SystemExit(1)
    else:
        print("All required input files are present.")


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    print("\n=== Model Playground: Loading Data ===")
    f_df = cast(
        pd.DataFrame,
        pd.read_csv(filepath_or_buffer=MODELS_FEATURES_PATH),
    )
    l_df = cast(
        pd.DataFrame,
        pd.read_csv(filepath_or_buffer=MODELS_LABELS_PATH),
    )
    print(f"Feature file: {MODELS_FEATURES_PATH.relative_to(PROJECT_ROOT)}")
    print(f"Labels file: {MODELS_LABELS_PATH.relative_to(PROJECT_ROOT)}")
    return f_df, l_df


def validate_data_preconditions(f_df: pd.DataFrame, l_df: pd.DataFrame):
    errors = []

    if f_df.empty:
        errors.append("feature dataset is empty")
    if l_df.empty:
        errors.append("label dataset is empty")
    if f_df.shape[0] != l_df.shape[0]:
        errors.append(
            f"row count mismatch (features={f_df.shape[0]}, labels={l_df.shape[0]})"
        )
    if ANOMALY_FLAG_COLUMN not in l_df.columns:
        errors.append(
            f"label dataset must contain '{ANOMALY_FLAG_COLUMN}' column"
        )

    if errors:
        print("Error: invalid input datasets:")
        for error in errors:
            print(f"\t- {error}")
        raise SystemExit(1)


def inspect_data(f_df: pd.DataFrame, l_df: pd.DataFrame):
    print("\n=== Model Playground: Data Inspection ===")
    print(f"Number of samples: {f_df.shape[0]}")
    print(f"Number of features: {f_df.shape[1]}")
    print(f"Features columns: {f_df.columns.tolist()}")
    print(
        f"Target values detected: "
        f"{sorted(l_df[ANOMALY_FLAG_COLUMN].unique().tolist())}"
    )


def prepare_features_and_labels(f_df: pd.DataFrame, l_df: pd.DataFrame):
    print("\n=== Model Playground: Preparing Features and Labels ===")
    X = f_df.values
    y = l_df[ANOMALY_FLAG_COLUMN].values
    print(f"X shape: {X.shape}")
    print(f"y shape: {y.shape}")
    return X, y


def split_data(X, y):
    print("\n=== Model Playground: Train/Test splitting ===")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")

    return X_train, X_test, y_train, y_test


def define_models():
    print("\n=== Model Playground: Loading Models ===")
    models = {
        "Decision Tree (baseline)": DecisionTreeClassifier(random_state=42),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Random Forest": RandomForestClassifier(random_state=42),
    }

    print(f"Models:\n{pformat(models, sort_dicts=False)}")

    return models


def train_models(models: dict, X_train, y_train, verbose: bool = True):
    if verbose:
        print("\n=== Model Playground: Training Models ===")
    trained_models = {}
    for name, model in models.items():
        if verbose:
            print(f"\nTraining {name}...")
        model.fit(X_train, y_train)
        if verbose:
            print(f"{name} training completed.")
        trained_models[name] = model

    return trained_models


def generate_predictions(trained_models: dict, X_test, verbose: bool = True):
    if verbose:
        print("\n=== Model Playground: Generating Predictions ===")
    prediction_results = []
    for name, model in trained_models.items():
        if verbose:
            print(f"Generating predictions with {name}...")
        y_pred = model.predict(X_test)
        prediction_results.append({"name": name, "y_pred": y_pred})
        if verbose:
            print(f"{name} predictions generated.")

    return prediction_results


def print_example_predictions(prediction_results, y_test, num_examples=5):
    sample_count = min(num_examples, len(y_test))
    print(
        "\n=== Model Playground: "
        f"Example Predictions (first {sample_count} samples) ==="
    )

    if sample_count == 0:
        print("No test samples available to display.")
        return

    for i in range(sample_count):
        line = f"True: {y_test[i]}"
        for result in prediction_results:
            model_name = result["name"]
            y_pred = result["y_pred"]
            line += f" | {model_name}: {y_pred[i]}"
        print(line)

def compute_accuracy(prediction_results, y_test, verbose: bool = True):
    if verbose:
        print("\n=== Model Playground: Accuracy Scores ===")
    for result in prediction_results:
        model_name = result["name"]
        y_pred = result["y_pred"]
        accuracy = (y_pred == y_test).mean()
        result["accuracy"] = accuracy
        if verbose:
            print(f"{model_name} Accuracy: {accuracy:.4f}")
    return prediction_results


def compute_detailed_metrics(prediction_results, y_test):
    print("\n=== Model Playground: Detailed Evaluation ===")
    for result in prediction_results:
        y_pred = result["y_pred"]
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)

        result["confusion_matrix"] = cm
        result["classification_report"] = report

        accuracy = result.get("accuracy", (y_pred == y_test).mean())
        normal_report = report.get("0", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        anomaly_report = report.get("1", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        macro_avg = report.get("macro avg", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})
        weighted_avg = report.get("weighted avg", {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0})

        print(f"\nModel: {result['name']}")
        print(f"Accuracy: {accuracy:.4f}")
        print("\nConfusion Matrix:")
        print(result["confusion_matrix"])
        print("\nClass labels:")
        print("0 -> normal observation")
        print("1 -> anomaly")
        print("\nClassification Report:")
        print("------------------------------------------------------------")
        print("Class Precision Recall F1-score Support")
        print("------------------------------------------------------------")
        print(
            f"0 (normal) "
            f"{normal_report['precision']:.2f} "
            f"{normal_report['recall']:.2f} "
            f"{normal_report['f1-score']:.2f} "
            f"{int(normal_report['support'])}"
        )
        print(
            f"1 (anomaly) "
            f"{anomaly_report['precision']:.2f} "
            f"{anomaly_report['recall']:.2f} "
            f"{anomaly_report['f1-score']:.2f} "
            f"{int(anomaly_report['support'])}"
        )
        print("------------------------------------------------------------")
        print(
            f"Macro average "
            f"{macro_avg['precision']:.2f} "
            f"{macro_avg['recall']:.2f} "
            f"{macro_avg['f1-score']:.2f} "
            f"{int(macro_avg['support'])}"
        )
        print(
            f"Weighted average "
            f"{weighted_avg['precision']:.2f} "
            f"{weighted_avg['recall']:.2f} "
            f"{weighted_avg['f1-score']:.2f} "
            f"{int(weighted_avg['support'])}"
        )

    return prediction_results


def rank_models(evaluation_results):
    print("\n=== Model Playground: Rank Models ===")
    sorted_results = sorted(
        evaluation_results,
        key=lambda result: result["accuracy"],
        reverse=True,
    )

    for i, result in enumerate(sorted_results, start=1):
        print(f"{i}. {result['name']} - {result['accuracy']:.4f}")

    return sorted_results


def save_experiment_summary(
    features_path,
    labels_path,
    X,
    X_train,
    X_test,
    ranked_models,
    experiment_results,
):
    print("\n=== Model Playground: Saving Summary ===")
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    def to_project_relative(path_like):
        path = Path(path_like)
        try:
            return path.relative_to(PROJECT_ROOT)
        except ValueError:
            return path

    num_samples = X.shape[0]
    num_features = X.shape[1]
    best_model = ranked_models[0] if ranked_models else None

    with open(MODEL_PLAYGROUND_SUMMARY_PATH, "w", encoding="utf-8") as f:
        f.write("OOAIS Model Playground Summary\n")
        f.write("=============================\n\n")

        f.write("Input datasets\n")
        f.write("--------------\n")
        f.write(f"{to_project_relative(features_path)}\n")
        f.write(f"{to_project_relative(labels_path)}\n\n")

        f.write("Dataset statistics\n")
        f.write("------------------\n")
        f.write(f"Number of samples: {num_samples}\n")
        f.write(f"Number of features: {num_features}\n")
        f.write(f"Training samples: {len(X_train)}\n")
        f.write(f"Testing samples: {len(X_test)}\n\n")

        f.write("Compared models\n")
        f.write("---------------\n")
        if ranked_models:
            for result in ranked_models:
                f.write(f"- {result['name']}: {result['accuracy']:.4f}\n")
        else:
            f.write("- No model evaluation results available.\n")
        f.write("\n")

        f.write("Best model\n")
        f.write("----------\n")
        if best_model:
            f.write(
                f"{best_model['name']} achieved the highest accuracy: "
                f"{best_model['accuracy']:.4f}\n\n"
            )
        else:
            f.write("No best model can be selected because ranking is empty.\n\n")

        f.write("Selected detailed evaluation\n")
        f.write("----------------------------\n")
        if best_model:
            report = best_model.get("classification_report", {})
            anomaly_report = report.get(
                "1",
                {"precision": 0.0, "recall": 0.0, "f1-score": 0.0, "support": 0},
            )
            f.write(f"Model: {best_model['name']}\n")
            f.write(f"Accuracy: {best_model['accuracy']:.4f}\n")
            f.write(
                "Anomaly class (1) -> "
                f"precision: {anomaly_report['precision']:.4f}, "
                f"recall: {anomaly_report['recall']:.4f}, "
                f"f1-score: {anomaly_report['f1-score']:.4f}, "
                f"support: {int(anomaly_report['support'])}\n"
            )

            confusion = best_model.get("confusion_matrix")
            if confusion is not None:
                f.write("Confusion matrix:\n")
                for row in confusion:
                    f.write(f"{row.tolist()}\n")
            f.write("\n")
        else:
            f.write("Detailed evaluation is unavailable.\n\n")

        f.write("Additional experiments\n")
        f.write("----------------------\n")
        has_experiments = False
        for exp_group in experiment_results.values():
            for exp in exp_group:
                has_experiments = True
                f.write(f"- {exp['model_name']}: {exp['accuracy']:.4f}\n")
        if not has_experiments:
            f.write("- No controlled experiment results available.\n")
        f.write("\n")

        f.write("Conclusion\n")
        f.write("----------\n")
        if best_model:
            f.write(
                f"The best candidate for further experiments is {best_model['name']},\n"
            )
            f.write(
                "because it achieved the highest accuracy on the current test set.\n"
            )
        else:
            f.write(
                "Run model training and ranking first to identify the best candidate.\n"
            )

        f.write("\nGitHub Link: https://github.com/LilConsul/orbital-ai-labs.git\n")


    print(f"Saved file: {MODEL_PLAYGROUND_SUMMARY_PATH.relative_to(PROJECT_ROOT)}")


def create_metric_plots(ranked_models):
    print("\n=== Model Playground: Saving Visualizations ===")

    if not ranked_models:
        print("No ranked models available for plotting.")
        return

    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    model_names = [result["name"] for result in ranked_models]
    accuracies = [result.get("accuracy", 0.0) for result in ranked_models]

    def anomaly_metric(result, metric_name):
        report = result.get("classification_report", {})
        anomaly_report = report.get(
            "1",
            {"precision": 0.0, "recall": 0.0, "f1-score": 0.0},
        )
        return anomaly_report.get(metric_name, 0.0)

    precisions = [anomaly_metric(result, "precision") for result in ranked_models]
    recalls = [anomaly_metric(result, "recall") for result in ranked_models]
    f1_scores = [anomaly_metric(result, "f1-score") for result in ranked_models]

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    axes[0, 0].bar(model_names, accuracies)
    axes[0, 0].set_title("Accuracy")

    axes[0, 1].bar(model_names, precisions)
    axes[0, 1].set_title("Precision (Anomaly)")

    axes[1, 0].bar(model_names, recalls)
    axes[1, 0].set_title("Recall (Anomaly)")

    axes[1, 1].bar(model_names, f1_scores)
    axes[1, 1].set_title("F1-score (Anomaly)")

    for ax in axes.flat:
        ax.tick_params(axis="x", rotation=15)
        ax.set_ylabel("Score")
        ax.set_ylim(0.0, 1.0)

    plt.tight_layout()
    plt.savefig(MODEL_COMPARISON_PANEL_PATH)
    plt.close(fig)

    print(f"Saved file: {MODEL_COMPARISON_PANEL_PATH.relative_to(PROJECT_ROOT)}")


def run_controlled_experiments(X_train, X_test, y_train, y_test):
    print("\n=== Model Playground: Controlled Experiments ===")

    experiment_results = {
        "decision_tree_depth": [],
        "random_forest_size": [],
    }

    for depth in [2, 3, 5]:
        model_name = f"Decision Tree (max_depth={depth})"
        models = {
            model_name: DecisionTreeClassifier(max_depth=depth, random_state=42)
        }
        trained_models = train_models(models, X_train, y_train, verbose=False)
        prediction_results = generate_predictions(trained_models, X_test, verbose=False)
        prediction_results = compute_accuracy(prediction_results, y_test, verbose=False)

        accuracy = prediction_results[0]["accuracy"]
        print(f"{model_name}: {accuracy:.2f}")
        experiment_results["decision_tree_depth"].append(
            {"parameter_value": depth, "model_name": model_name, "accuracy": accuracy}
        )

    for n_estimators in [5, 10, 50]:
        model_name = f"Random Forest (n_estimators={n_estimators})"
        models = {
            model_name: RandomForestClassifier(
                n_estimators=n_estimators,
                random_state=42,
            )
        }
        trained_models = train_models(models, X_train, y_train, verbose=False)
        prediction_results = generate_predictions(trained_models, X_test, verbose=False)
        prediction_results = compute_accuracy(prediction_results, y_test, verbose=False)

        accuracy = prediction_results[0]["accuracy"]
        print(f"{model_name}: {accuracy:.2f}")
        experiment_results["random_forest_size"].append(
            {
                "parameter_value": n_estimators,
                "model_name": model_name,
                "accuracy": accuracy,
            }
        )

    return experiment_results

def plot_results(experiment_results, show_plot: bool = True):
    print("\n=== Model Playground: Plot Results ===")

    dt_results = experiment_results.get("decision_tree_depth", [])
    rf_results = experiment_results.get("random_forest_size", [])

    if not dt_results and not rf_results:
        print("No experiment results available to plot.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    if dt_results:
        dt_x = [item["parameter_value"] for item in dt_results]
        dt_y = [item["accuracy"] for item in dt_results]
        axes[0].plot(dt_x, dt_y, marker="o")
        axes[0].set_title("Decision Tree Depth")
        axes[0].set_xlabel("max_depth")
        axes[0].set_ylabel("accuracy")
        axes[0].set_ylim(0.0, 1.0)
    else:
        axes[0].set_title("Decision Tree Depth")
        axes[0].text(0.5, 0.5, "No data", ha="center", va="center")
        axes[0].set_axis_off()

    if rf_results:
        rf_x = [item["parameter_value"] for item in rf_results]
        rf_y = [item["accuracy"] for item in rf_results]
        axes[1].plot(rf_x, rf_y, marker="o", color="tab:green")
        axes[1].set_title("Random Forest Size")
        axes[1].set_xlabel("n_estimators")
        axes[1].set_ylabel("accuracy")
        axes[1].set_ylim(0.0, 1.0)
    else:
        axes[1].set_title("Random Forest Size")
        axes[1].text(0.5, 0.5, "No data", ha="center", va="center")
        axes[1].set_axis_off()

    fig.tight_layout()

    if show_plot:
        plt.show()
    else:
        plt.close(fig)

if __name__ == "__main__":
    validate_input_files()

    features_df, labels_df = load_data()
    validate_data_preconditions(features_df, labels_df)

    inspect_data(features_df, labels_df)

    X, y = prepare_features_and_labels(features_df, labels_df)
    X_train, X_test, y_train, y_test = split_data(X, y)

    models = define_models()
    trained_models = train_models(models, X_train, y_train)

    prediction_results = generate_predictions(trained_models, X_test)
    print_example_predictions(prediction_results, y_test)
    prediction_results = compute_accuracy(prediction_results, y_test)
    prediction_results = compute_detailed_metrics(prediction_results, y_test)
    sorted_prediction_results = rank_models(prediction_results)

    create_metric_plots(sorted_prediction_results)

    experiment_results = run_controlled_experiments(X_train, X_test, y_train, y_test)
    plot_results(experiment_results, show_plot=False)
    save_experiment_summary(
        features_path=MODELS_FEATURES_PATH,
        labels_path=MODELS_LABELS_PATH,
        X=X,
        X_train=X_train,
        X_test=X_test,
        ranked_models=sorted_prediction_results,
        experiment_results=experiment_results,
    )

