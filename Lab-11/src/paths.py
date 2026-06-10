from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_DIR / "data"
IMAGE_DIR = DATA_DIR / "segmentation" / "images"
MASK_DIR = DATA_DIR / "segmentation" / "masks"
REPORT_DIR = PROJECT_DIR / "reports" / "segmentation_examples"
REPORT_DIR_HARD = PROJECT_DIR / "reports" / "segmentation_examples_hard"
REPORT_DIR_CLASSES = PROJECT_DIR / "reports" / "segmentation_examples_classes"
MODEL_DIR = PROJECT_DIR / "models"


def make_dir(paths: list):
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def get_image_path(directory: Path, img_num=0):
    return sorted(directory.glob("*.png"))[img_num]


make_dir([DATA_DIR, IMAGE_DIR, MASK_DIR, REPORT_DIR, MODEL_DIR, REPORT_DIR_HARD, REPORT_DIR_CLASSES])
