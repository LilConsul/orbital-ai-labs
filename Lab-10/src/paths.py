from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_DIR / "models"
REPORTS_DIR = PROJECT_DIR / "reports"
IMAGES_DIR = PROJECT_DIR / "data" / "processed" / "images"
INTERFERENCE_SAMPLE = PROJECT_DIR / "data" / "inference_samples" / "noise.jpg"
HIGHWAY_PATH = (
    PROJECT_DIR / "data" / "raw" / "eurosat" / "2750" / "Highway" / "Highway_1.jpg"
)
TEST_DIR = IMAGES_DIR / "test"
TRAIN_DIR = IMAGES_DIR / "train"
GRADCAM_EXAMPLES_DIR = REPORTS_DIR / "gradcam_examples"


REPORTS_DIR.mkdir(parents=True, exist_ok=True)

MODELS_TRANSFER_MODEL_PATH = MODELS_DIR / "resnet18_transfer.pt"
MODELS_TRANSFER_CLASS_PATH = MODELS_DIR / "resnet18_transfer.txt"


def get_image_path(directory: Path, f_num=0, img_num=0):
    folder = [p for p in directory.iterdir() if p.is_dir()][f_num]
    image_path = list(folder.glob("*.jpg"))[img_num]
    return image_path
