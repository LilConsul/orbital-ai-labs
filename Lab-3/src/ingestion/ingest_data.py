import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
DATASET_PATH = RAW_DATA_DIR / "orbital_observations.csv"
METADATA_PATH = RAW_DATA_DIR / "metadata.json"

PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)


with open(METADATA_PATH, "r", encoding="utf-8") as metadata_file:
    metadata = json.load(metadata_file)

with open(DATASET_PATH, "r", encoding="utf-8", newline="") as dataset_file:
    dataset_reader = csv.DictReader(dataset_file)
    dataset_columns = dataset_reader.fieldnames or []
    rows = list(dataset_reader)

print(f"Dataset: {metadata['dataset_name']}")
print(f"Records loaded: {len(rows)}")
# print(f"Columns (dataset): {dataset_columns}")
# print(f"Columns (metadata): {metadata['columns']}")

if dataset_columns == metadata["columns"]:
    print("Columns validation: OK")
    print(f"Columns: {dataset_columns}")

else:
    print("Columns validation: MISMATCH")
    print(f"Columns (metadata): {metadata['columns']}")
    missing_in_dataset = set(metadata["columns"]) - set(dataset_columns)
    extra_in_dataset = set(dataset_columns) - set(metadata["columns"])
    if missing_in_dataset:
        print(f"Missing columns in dataset: {missing_in_dataset}")
    if extra_in_dataset:
        print(f"Extra columns in dataset: {extra_in_dataset}")

if len(rows) == metadata["num_records"]:
    print("Record count: OK")

else:
    print("Record count: MISMATCH")
    print(f"Expected: {metadata['num_records']:^2}")
    print(f"Actual: {len(rows):^7}")

valid_records = []
invalid_records = []
model_input = []

for row in rows:
    if any("INVALID" in str(value) for value in row.values()):
        invalid_records.append(row)
        continue
    valid_records.append(row)
    model_input.append({col: row[col] for col in metadata["feature_columns"]})

print(f"Valid records: {len(valid_records)}")
print(
    f"Valid records sample (first 3 lines):\n{json.dumps(valid_records[:3], indent=2)}"
)
print(f"Invalid records: {len(invalid_records)}")


def write_to_file(filename: Path, records: list, columns: list):
    try:
        with open(filename, "w", encoding="utf-8", newline="") as data_file:
            dataset_writer = csv.DictWriter(data_file, fieldnames=columns)
            dataset_writer.writeheader()
            dataset_writer.writerows(records)
    except Exception as e:
        print(f"Exception while writing to file occurred:\n{e}")
    else:
        print(f"File written: {filename} ({len(records)} records)")


write_to_file(
    PROCESSED_DATA_DIR / "valid_observations.csv", valid_records, dataset_columns
)
write_to_file(
    PROCESSED_DATA_DIR / "invalid_observations.csv", invalid_records, dataset_columns
)

print(f"Model input sample (first 3 records):\n{json.dumps(model_input[:3], indent=2)}")

write_to_file(
    PROCESSED_DATA_DIR / "model_input.csv", model_input, metadata["feature_columns"]
)
