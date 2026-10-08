"""
Image_arithmetic.py  -  Arithmetic operations on two GRAYSCALE images.

Put your two images next to this file and name them:
    input_image1.jpg   (image A)
    input_image2.jpg   (image B)
or pass paths:  python Image_arithmetic.py a.png b.png
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import arithmetic_core as ac

IMAGE_A = sys.argv[1] if len(sys.argv) > 1 else "input_image1.jpg"
IMAGE_B = sys.argv[2] if len(sys.argv) > 2 else "input_image2.jpg"
RESULTS_DIR = "results"
ALPHA = 0.7


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    for p in (IMAGE_A, IMAGE_B):
        if not os.path.exists(p):
            sys.exit(f"[!] Image '{p}' not found. Place it in this folder.")

    a, b = ac.load_pair(IMAGE_A, IMAGE_B, "L")
    print(f"Image A: {IMAGE_A}  |  Image B: {IMAGE_B} (resized to {a.shape[1]} x {a.shape[0]})")

    ops = ac.all_operations(a, b, ALPHA)

    # Save every result as its own PNG + a small statistics table
    lines = ["Operation                         Mean    Min   Max"]
    for stem, (title, res) in ops.items():
        Image.fromarray(res).save(os.path.join(RESULTS_DIR, f"{stem}.png"))
        lines.append(f"{title:<30} {res.mean():7.2f} {res.min():5d} {res.max():5d}")
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    with open(os.path.join(RESULTS_DIR, "stats.txt"), "w") as f:
        f.write(report)

    # One overview figure
    panels = [("Image A", a), ("Image B", b)] + [(t, r) for t, r in ops.values()]
    cols = 5
    rows = int(np.ceil(len(panels) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3.4 * cols, 3.4 * rows))
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (title, img) in zip(axes.ravel(), panels):
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)
        ax.set_title(title, fontsize=10)
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "arithmetic_gray_overview.png"), dpi=130)
    plt.close()

    print(f"Results saved in '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()
