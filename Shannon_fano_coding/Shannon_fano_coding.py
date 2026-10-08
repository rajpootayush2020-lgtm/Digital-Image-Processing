"""
Shannon_fano_coding.py  -  Grayscale image compression with Shannon-Fano coding.

Put your image next to this file and name it:  input_image.jpg
(or pass a different path:  python Shannon_fano_coding.py my_photo.png)
"""

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

import shannon_fano_core as sf

INPUT_IMAGE = sys.argv[1] if len(sys.argv) > 1 else "input_image.jpg"
RESULTS_DIR = "results"


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)

    if not os.path.exists(INPUT_IMAGE):
        sys.exit(f"[!] Image '{INPUT_IMAGE}' not found. Place it in this folder.")

    # 1. Load as grayscale
    img = np.array(Image.open(INPUT_IMAGE).convert("L"), dtype=np.uint8)
    h, w = img.shape
    flat = img.flatten()
    print(f"Loaded '{INPUT_IMAGE}'  ->  {w} x {h} pixels")

    # 2. Encode and save
    codes, payload, nbits = sf.encode_channel(flat)
    bin_path = os.path.join(RESULTS_DIR, "compressed_gray.bin")
    sf.write_bin(bin_path, h, w, [(codes, payload, nbits)])

    # 3. Decode from file and verify
    h2, w2, chans = sf.read_bin(bin_path)
    rec = sf.decode_channel(*chans[0], count=h2 * w2).reshape(h2, w2)
    lossless = np.array_equal(img, rec)
    Image.fromarray(rec).save(os.path.join(RESULTS_DIR, "reconstructed_gray.png"))

    # 4. Code table (most frequent first)
    counts = np.bincount(flat, minlength=256)
    with open(os.path.join(RESULTS_DIR, "code_table_gray.txt"), "w") as f:
        f.write("Pixel  Count      Code\n")
        for s in sorted(codes, key=lambda s: -counts[s]):
            f.write(f"{s:<6} {counts[s]:<10} {codes[s]}\n")

    # 5. Statistics
    original_bytes = flat.size
    compressed_bytes = os.path.getsize(bin_path)
    ent = sf.entropy(flat)
    avg = sf.average_length(flat, codes)

    report = (
        f"Image size            : {w} x {h}\n"
        f"Unique gray levels    : {len(codes)}\n"
        f"Original size         : {original_bytes} bytes\n"
        f"Compressed size       : {compressed_bytes} bytes\n"
        f"Compression ratio     : {original_bytes / compressed_bytes:.3f} : 1\n"
        f"Space saved           : {100 * (1 - compressed_bytes / original_bytes):.2f} %\n"
        f"Entropy               : {ent:.4f} bits/pixel\n"
        f"Average code length   : {avg:.4f} bits/pixel\n"
        f"Coding efficiency     : {100 * ent / avg:.2f} %\n"
        f"Lossless (verified)   : {lossless}\n"
    )
    print("\n" + report)
    with open(os.path.join(RESULTS_DIR, "stats_gray.txt"), "w") as f:
        f.write(report)

    # 6. Plots
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.5))
    ax[0].imshow(img, cmap="gray", vmin=0, vmax=255)
    ax[0].set_title("Original (grayscale)")
    ax[0].axis("off")

    ax[1].imshow(rec, cmap="gray", vmin=0, vmax=255)
    ax[1].set_title(f"Reconstructed (lossless = {lossless})")
    ax[1].axis("off")

    ax[2].bar(range(256), counts, color="seagreen", width=1.0)
    ax[2].set_title("Histogram of gray levels")
    ax[2].set_xlabel("Pixel value")
    ax[2].set_ylabel("Count")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "shannon_fano_gray_result.png"), dpi=150)
    plt.close()

    syms = sorted(codes)
    plt.figure(figsize=(9, 4))
    plt.bar(syms, [len(codes[s]) for s in syms], color="purple", width=1.0)
    plt.title("Shannon-Fano code length per gray level")
    plt.xlabel("Pixel value")
    plt.ylabel("Code length (bits)")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, "code_lengths_gray.png"), dpi=150)
    plt.close()

    print(f"Results saved in '{RESULTS_DIR}/'")


if __name__ == "__main__":
    main()
