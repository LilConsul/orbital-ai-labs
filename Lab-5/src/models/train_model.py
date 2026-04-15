import csv
from pathlib import Path

import joblib
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "model_features.csv"
LABELS_PATH = PROCESSED_DIR / "model_labels.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
REPORTS_DIR = PROJECT_ROOT / "reports"
MODEL_PATH = RESULTS_DIR / "model_trained.joblib"
EVALUATION_RESULTS_PATH = RESULTS_DIR / "evaluation_results.txt"
TRAINING_REPORT_PATH = REPORTS_DIR / "training_report.txt"

print("=== Machine Learning: Loading Feature Dataset ===")
print(f"Input file: {FEATURES_PATH.relative_to(PROJECT_ROOT)}")
print(f"Labels file: {LABELS_PATH.relative_to(PROJECT_ROOT)}")
with open(FEATURES_PATH, "r", encoding="utf-8", newline="") as dataset_file:
    dataset_reader = csv.DictReader(dataset_file)
    column_names = dataset_reader.fieldnames or []
    rows = list(dataset_reader)

with open(LABELS_PATH, "r", encoding="utf-8", newline="") as dataset_file:
    dataset_reader = csv.DictReader(dataset_file)
    label_column_names = dataset_reader.fieldnames or []
    label_rows = list(dataset_reader)

print(f"Records loaded: {len(rows)}")
print(f"Columns loaded: {column_names}")
print(f"Features loaded: {len(label_rows)}")
print(f"Features names: {label_column_names}")

print("\n=== Machine Learning: Preparing Features and Target ===")
X = [[float(row[column]) for column in column_names] for row in rows]

label_column = "anomaly_flag"
if label_column not in label_column_names and label_column_names:
    label_column = label_column_names[0]

y = [int(row[label_column]) for row in label_rows]

if len(X) != len(y):
    raise ValueError(f"Mismatched sample counts: len(X)={len(X)} and len(y)={len(y)}")

print(f"Number of samples in X: {len(X)}")
print(f"Number of labels in y: {len(y)}")
print(f"Target values detected: {sorted(set(y))}")

print("\n=== Machine Learning: Train/Test Split ===")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")

print("\n=== Machine Learning: Model Training ===")

model = DecisionTreeClassifier()
print(f"Model: {model.__class__.__name__}")

model.fit(X=X_train, y=y_train)
print("Training completed successfully.")

print("\n=== Machine Learning: Prediction ===")
x_pred = model.predict(X_test)
print("Predictions generated successfully.")
print(f"Number of predictions: {len(x_pred)}")
print(f"\nPredicted values (first 10): {x_pred[:10]}")

print("\n=== Machine Learning: Evaluation ===")
accuracy = accuracy_score(y_true=y_test, y_pred=x_pred)
conf_matrix = confusion_matrix(y_true=y_test, y_pred=x_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Confusion matrix:\n{conf_matrix}")

print("\n=== Machine Learning: Saving and Inspecting Model ===")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
joblib.dump(model, MODEL_PATH)
print(f"Saved model: {MODEL_PATH.relative_to(PROJECT_ROOT)}")

print(f"Model type: {model.__class__.__name__}")
print(f"Tree depth: {model.get_depth()}")
print(f"Number of leaves: {model.get_n_leaves()}")

tree_rules = export_text(model, feature_names=column_names)
print(f"\nDecision Tree Rules:\n{tree_rules}")

print("\n=== Machine Learning: Saving Evaluation Results ===")
with open(EVALUATION_RESULTS_PATH, "w", encoding="utf-8") as results_file:
    results_file.write(
        f"""OOAIS Model Evaluation
{"=" * 40}

=== Model Evaluation Results ===
Model name: {model.__class__.__name__}
Number of training samples: {len(X_train)}
Number of testing samples: {len(X_test)}

Accuracy: {accuracy:.4f}
Confusion matrix:\n{conf_matrix}

=== Tree Representation ===
Tree depth: {model.get_depth()}
Number of leaves: {model.get_n_leaves()}
Decision Tree Rules:\n{tree_rules}
"""
    )

print(f"Saved evaluation results: {EVALUATION_RESULTS_PATH.relative_to(PROJECT_ROOT)}")

print("\n=== Machine Learning: Saving Training Report ===")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
with open(TRAINING_REPORT_PATH, "w", encoding="utf-8") as report_file:
    report_file.write(
        f"""OOAIS Model Training Report
{"=" * 40}

Input datasets
--------------
{FEATURES_PATH.relative_to(PROJECT_ROOT)}
{LABELS_PATH.relative_to(PROJECT_ROOT)} 

Dataset statistics
------------------
Number of samples: {len(X)}
Number of features: {len(column_names)}

Model
-----
{model.__class__.__name__}

Train/Test split
----------------
Training samples: {len(X_train)}
Test samples: {len(X_test)}

Evaluation summary
------------------
Accuracy: {accuracy:.4f}
Confusion Matrix:
{conf_matrix}

GitHub
------------------
GitHub Repostiroy Link: https://github.com/LilConsul/orbital-ai-labs.git
"""
    )

print(f"Saved training report: {TRAINING_REPORT_PATH.relative_to(PROJECT_ROOT)}")
