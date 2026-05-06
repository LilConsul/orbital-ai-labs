from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "images"
TRAIN_DIR = PROCESSED_DIR / "train"
TEST_DIR = PROCESSED_DIR / "test"
BATCH_SIZE = 16

MODEL_PATH = PROJECT_ROOT / "models" / "cnn_model.pt"
CLASS_NAMES_PATH = PROJECT_ROOT / "models" / "cnn_classes.txt"
EPOCHS = 30
LEARNING_RATE = 0.001

REPORT_DIR = PROJECT_ROOT / "reports"
CONFUSION_MATRIX_PATH = REPORT_DIR / "confusion_matrix-aug-epoch30-relu3.png"
