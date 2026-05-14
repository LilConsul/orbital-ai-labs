from pathlib import Path
import torch
from torch import nn
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt

from paths import MODELS_DIR, REPORT_DIR

MODEL_PATH = MODELS_DIR / "resnet18_transfer.pt"
CLASSES_PATH = MODELS_DIR / "resnet18_transfer_classes.txt"

OUTPUT_PATH = REPORT_DIR / "transfer_predictions_gallery.png"
IMAGES_PER_CLASS = 3

# PAGE 27 TASK 2