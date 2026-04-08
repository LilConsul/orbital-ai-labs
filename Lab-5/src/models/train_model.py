import csv
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
FEATURES_PATH = PROCESSED_DIR / "model_features.csv"
LABELS_PATH = PROCESSED_DIR / "model_labels.csv"

print("=== Machine Learning: Loading Feature Dataset ===")
print(f"Input file: {FEATURES_PATH}")
with open(FEATURES_PATH, "r", encoding="utf-8", newline="") as dataset_file:
    dataset_reader = csv.DictReader(dataset_file)
    column_names = dataset_reader.fieldnames
    rows = list(dataset_reader)

with open(LABELS_PATH, "r", encoding="utf-8", newline="") as dataset_file:
    dataset_reader = csv.DictReader(dataset_file)
    feature_names = dataset_reader.fieldnames
    features = list(dataset_reader)

print(f"Records loaded: {len(rows)}")
print(f"Columns loaded: {column_names}")
print(f"Features loaded: {len(features)}")
print(f"Features names: {feature_names}")
