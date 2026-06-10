import random

import numpy as np
from paths import IMAGE_DIR, MASK_DIR, PROJECT_DIR
from PIL import Image, ImageDraw

CLASS_COLORS = {
    0: (80, 80, 80),  # background
    1: (40, 140, 40),  # vegetation
    2: (40, 80, 180),  # water
    3: (180, 180, 180),  # urban
}
IMAGE_SIZE = 128
NUM_IMAGES = 200
RANDOM_SEED = 42




def vary_color(color):
    return color


def create_scene(index):
    image = Image.new("RGB", (IMAGE_SIZE, IMAGE_SIZE), CLASS_COLORS[0])
    mask = Image.new("L", (IMAGE_SIZE, IMAGE_SIZE), 0)

    image_draw = ImageDraw.Draw(image)
    mask_draw = ImageDraw.Draw(mask)

    # Vegetation (irregular polygon)
    veg_points = [
        (random.randint(0, IMAGE_SIZE), random.randint(0, IMAGE_SIZE))
        for _ in range(random.randint(8, 15))
    ]

    image_draw.polygon(veg_points, fill=vary_color(CLASS_COLORS[1]))
    mask_draw.polygon(veg_points, fill=1)

    # Water (curved river)
    river_x = random.randint(30, 90)
    river_points = []

    x = river_x
    for y in range(0, IMAGE_SIZE, 6):
        x += random.randint(-7, 7)
        x = max(5, min(IMAGE_SIZE - 5, x))
        river_points.append((x, y))

    image_draw.line(
        river_points, fill=vary_color(CLASS_COLORS[2]), width=random.randint(5, 12)
    )

    mask_draw.line(river_points, fill=2, width=random.randint(5, 12))

    # Urban areas (variable rectangles + density)
    for _ in range(random.randint(8, 18)):
        ux = random.randint(0, IMAGE_SIZE - 20)
        uy = random.randint(0, IMAGE_SIZE - 20)

        w = random.randint(5, 25)
        h = random.randint(5, 25)

        image_draw.rectangle([ux, uy, ux + w, uy + h], fill=vary_color(CLASS_COLORS[3]))
        mask_draw.rectangle([ux, uy, ux + w, uy + h], fill=3)

    # Small random noise objects (mixed classes)
    for _ in range(60):
        x = random.randint(0, IMAGE_SIZE - 3)
        y = random.randint(0, IMAGE_SIZE - 3)
        cls = random.randint(1, 3)

        image_draw.rectangle([x, y, x + 2, y + 2], fill=vary_color(CLASS_COLORS[cls]))
        mask_draw.rectangle([x, y, x + 2, y + 2], fill=cls)

    # Global noise (stronger, realistic)
    image_array = np.array(image).astype(np.int16)
    noise = np.random.normal(0, 25, image_array.shape)
    image_array = np.clip(image_array + noise, 0, 255).astype(np.uint8)

    image = Image.fromarray(image_array)

    # Random rotation (important realism boost)
    angle = random.uniform(0, 360)

    image = image.rotate(angle)
    mask = mask.rotate(angle, resample=Image.NEAREST)

    # Save
    image.save(IMAGE_DIR / f"scene_{index:04d}.png")
    mask.save(MASK_DIR / f"scene_{index:04d}.png")


def main():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)
    MASK_DIR.mkdir(parents=True, exist_ok=True)
    for index in range(NUM_IMAGES):
        create_scene(index)

    print("=== Synthetic Segmentation Dataset ===")
    print(f"Generated images: {NUM_IMAGES}")
    print(f"Image folder: {IMAGE_DIR.relative_to(PROJECT_DIR)}")
    print(f"Mask folder: {MASK_DIR.relative_to(PROJECT_DIR)}")


if __name__ == "__main__":
    main()
