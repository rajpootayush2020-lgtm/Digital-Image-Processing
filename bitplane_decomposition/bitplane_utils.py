from pathlib import Path
import numpy as np
from PIL import Image


def load_image(filename):
    return Image.open(filename).convert("RGB")


def extract_planes(channel):
    values = np.asarray(channel, dtype=np.uint8)
    return [((values >> bit) & 1).astype(np.uint8) * 255 for bit in range(8)]


def save_gray_planes(image, destination):
    destination.mkdir(parents=True, exist_ok=True)
    planes = extract_planes(image.convert("L"))

    for bit, plane in enumerate(planes):
        Image.fromarray(plane).save(destination / f"gray_plane_{bit}.png")

    return planes


def save_rgb_planes(image, destination):
    destination.mkdir(parents=True, exist_ok=True)
    data = np.asarray(image, dtype=np.uint8)

    for index, name in enumerate(("red", "green", "blue")):
        for bit, plane in enumerate(extract_planes(data[:, :, index])):
            Image.fromarray(plane).save(
                destination / f"{name}_plane_{bit}.png"
            )


def make_montage(planes, output_file, columns=4):
    if not planes:
        return

    height, width = planes[0].shape
    rows = (len(planes) + columns - 1) // columns
    canvas = np.zeros((rows * height, columns * width), dtype=np.uint8)

    for index, plane in enumerate(planes):
        row, col = divmod(index, columns)
        canvas[
            row * height:(row + 1) * height,
            col * width:(col + 1) * width
        ] = plane

    Image.fromarray(canvas).save(output_file)
