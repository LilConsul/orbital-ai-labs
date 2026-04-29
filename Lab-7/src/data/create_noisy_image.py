import numpy as np
from PIL import Image
from src.paths import DATA_ROOT

output_path = DATA_ROOT /"inference_samples"/"noise.jpg"
output_path.parent.mkdir(parents=True, exist_ok=True)
array = np.random.randint(0, 256, size=(64, 64, 3), dtype=np.uint8)
image = Image.fromarray(array)
image.save(output_path)
print(f"Saved noise image: {output_path}")
