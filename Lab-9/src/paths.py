from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed" / "images"
TRAIN_DIR = PROCESSED_DIR / "train"
TEST_DIR = PROCESSED_DIR / "test"

REPORT_PATH = PROJECT_ROOT / "reports" / "resnet18_transfer.txt"

MODELS_DIR = PROJECT_ROOT / "models"
MODEL_PATH = MODELS_DIR / "resnet18_transfer.pt"
CLASS_NAMES_PATH = MODELS_DIR / "cnn_classes.txt"

EPOCHS = 8
LEARNING_RATE = 0.001
BATCH_SIZE = 16

REPORT_DIR = PROJECT_ROOT / "reports"
CONFUSION_MATRIX_PATH = REPORT_DIR / "transfer_confusion_matrix.png"

NOISE_IMAGE_PATH = DATA_DIR / "inference_samples" / "noise.jpg"
