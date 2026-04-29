import numpy as np


def extract_features(image):
    image = image.convert("RGB")
    image = image.resize((64, 64))
    array = np.array(image) / 255.0

    return array.flatten()
