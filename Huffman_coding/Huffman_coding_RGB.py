"""
Huffman_coding_RGB.py  -  Colour (RGB) image compression, one Huffman code per channel.

Put your image next to this file and name it:  input_image.jpg
(or pass a different path:  python Huffman_coding_RGB.py my_photo.png)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import huffman_core as hc

INPUT_IMAGE = sys.argv[1] if len(sys.argv) > 1 else "input_image.jpg"
RESULTS_DIR = "results"
NAMES = ["Red", "Green", "Blue"]
COLORS = ["red", "green", "blue"]


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(INPUT_IMAGE):
        sys.exit(f"[!] Image '{INPUT_IMAGE}' not found. Place it in this folder.")

    img = np.array(Image.open(INPUT_IMAGE).convert("RGB"), dtype=np.uint8)
    h, w, _ = img.shape
    print(f"Loaded '{INPUT_IMAGE}'  ->  {w} x {h} pixels (RGB)")

    # Encode each channel separately
    encoded, all_lengths = [], []
    for c in range(3):
        lengths, payload, nbits = hc.encode_channel(img[:, :, c].flatten())
        encoded.append((lengths, payload, nbits))
        all_lengths.append(lengths)

    bin_path = os.path.join(RESULTS_DIR, "compressed_rgb.bin")
    hc.write_bin(bin_path, h, w, encoded)

    # Decode + verify
    h2, w2, chans = hc.read_bin(bin_path)
    rec = np.stack(
        [hc.decode_channel(*chans[c], count=h2 * w2).reshape(h2, w2) for c in range(3)],
        axis=2,
    )
    lossless = np.array_equal(img, rec)
    Image.fromarray(rec).save(os.path.join(RESULTS_DIR, "reconstructed_rgb.png"))

    # Stats
    original_bytes = img.size
    compressed_bytes = os.path.getsize(bin_path)
    lines = [
        f"Image size            : {w} x {h}",
        f"Original size         : {original_bytes} bytes",
        f"Compressed size       : {compressed_bytes} bytes",
        f"Compression ratio     : {original_bytes / compressed_bytes:.3f} : 1",
        f"Space saved           : {100 * (1 - compressed_bytes / original_bytes):.2f} %",
        f"Lossless (verified)   : {lossless}",
        "",
    ]
    for c in range(3):
        flat = img[:, :, c].flatten()
        lines.append(
            f"{NAMES[c]:<5} channel -> entropy {hc.entropy(flat):.4f} bpp | "
            f"avg code length {hc.average_length(flat, all_lengths[c]):.4f} bpp"
        )
    report = "\n".join(lines) + "\n"
    print("\n" + report)
    with open(os.path.join(RESULTS_DIR, "stats_rgb.txt"), "w") as f:
        f.write(report)

    # Plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].imshow(img)
    ax[0].set_title("Original")
    ax[0].axis("off")
    ax[1].imshow(rec)
    ax[1].set_title(f"Reconstructed (lossless = {lossless})")
    ax[1].axis("off")
    for c in range(3):
        ax[2].plot(np.bincount(img[:, :, c].flatten(), minlength=256),
                   color=COLORS[c], label=NAMES[c], linewidth=1)
    ax[2].set_title("RGB histograms")
    ax[2].set_xlabel("Pixel value")
    ax[2].set_ylabel("Count")
    ax[2].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "huffman_rgb_result.png"), dpi=150)
    plt.close()

    print(f"Results saved in '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()
