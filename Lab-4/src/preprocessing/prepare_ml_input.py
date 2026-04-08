import csv
import json
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
DATASET_PATH = DATA_DIR / "observations_valid.csv"

REQUIRED_NUMERIC_COLUMNS = ("temperature", "velocity", "altitude", "signal_strength")

print("=== ML Input Preparation: Loading and Conversion ===")

print(f"Input file: {DATASET_PATH}")
with open(DATASET_PATH, "r", encoding="utf-8", newline="") as dataset_file:
    dataset_reader = csv.DictReader(dataset_file)
    rows = list(dataset_reader)

accepted_count = 0
accepted_rows = []
rejected_count = 0

for row in rows:
    try:
        converted_row = dict(row)
        for column_name in REQUIRED_NUMERIC_COLUMNS:
            raw_value = row.get(column_name, "")
            if raw_value.strip() == "":
                raise ValueError
            converted_row[column_name] = float(raw_value)

        if converted_row["altitude"] < 0:
            raise ValueError

        accepted_count += 1
        accepted_rows.append(converted_row)
    except ValueError:
        rejected_count += 1

print(f"Records loaded: {len(rows)}")
print(f"Records accepted: {accepted_count}")
print(f"Records rejected: {rejected_count}")

print("\n=== ML Input Preparation: Normalization ===")

if accepted_rows:
    column_min_max = {}
    for column_name in REQUIRED_NUMERIC_COLUMNS:
        column_values = [row[column_name] for row in accepted_rows]
        column_min_max[column_name] = (min(column_values), max(column_values))
    # print(column_min_max)

    for row in accepted_rows:
        for column_name in REQUIRED_NUMERIC_COLUMNS:
            min_value, max_value = column_min_max[column_name]
            if max_value == min_value:
                row[column_name] = 0.0
            else:
                row[column_name] = (row[column_name] - min_value) / (
                    max_value - min_value
                )

print("Normalization completed successfully.")
print("All selected numerical features are in range [0,1].")
# print(accepted_rows[:5])

print("\n=== ML Input Preparation: Derived Features ===")
T_V_INTERACTION = "temperature_velocity_interaction"
A_S_RATIO = "altitude_signal_ratio"

for row in accepted_rows:
    row[T_V_INTERACTION] = row.get("temperature", 0.0) * row.get("velocity", 0.0)
    row[A_S_RATIO] = row.get("altitude", 0.0) / (
        row.get("signal_strength", 0.0) + 0.0001
    )

print(f"New features added: \n\t- {T_V_INTERACTION} \n\t- {A_S_RATIO}")
print(f"\nExample record (extended):\n{json.dumps(accepted_rows[0], indent=2)}")

print("\n=== ML Input Preparation: Temporal Features ===")

H_NORMALIZED = "hour_normalized"

for row in accepted_rows:
    timestamp = datetime.strptime(row.get("timestamp"), "%Y-%m-%d %H:%M:%S")
    row[H_NORMALIZED] = timestamp.hour / 24.0

print(f"New features added: \n\t- {H_NORMALIZED}")
print(f"\nExample record (extended):\n{json.dumps(accepted_rows[0], indent=2)}")

print("\n=== ML Input Preparation: Feature Selection ===")

SELECTED_COLUMNS = REQUIRED_NUMERIC_COLUMNS + (T_V_INTERACTION, A_S_RATIO, H_NORMALIZED)

print("Selected features:")
print("\n".join(f"\t- {col}" for col in SELECTED_COLUMNS))

print("\n=== ML Input Preparation: Feature Selection ===")

SELECTED_COLUMNS = REQUIRED_NUMERIC_COLUMNS + (T_V_INTERACTION, A_S_RATIO, H_NORMALIZED)

print("Selected features:")
print("\n".join(f"\t- {col}" for col in SELECTED_COLUMNS))

selected_rows = [
    {column_name: row[column_name] for column_name in SELECTED_COLUMNS}
    for row in accepted_rows
]
print(f"\nExample record (final):\n{json.dumps(selected_rows[0], indent=2)}")

print("\n=== ML Input Preparation: Saving Outputs ===")


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


anomalies = [{"anomaly_flag": row["anomaly_flag"]} for row in accepted_rows]

write_to_file(DATA_DIR / "model_features.csv", selected_rows, list(SELECTED_COLUMNS))
write_to_file(DATA_DIR / "model_labels.csv", anomalies, ["anomaly_flag"])

print(f"\nNumber of records: {len(selected_rows)}")
print(f"Number of features: {len(SELECTED_COLUMNS)}")
print(f"\nExample label record:\n{json.dumps(anomalies[0], indent=2)}")
