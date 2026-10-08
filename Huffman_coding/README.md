# 🗜️ Huffman Coding (Lossless Image Compression)

Lossless image compression using **canonical Huffman coding**.
Instead of storing the whole Huffman tree, only the *code length* of each gray
level is stored (256 bytes per channel); the decoder rebuilds identical codes
from those lengths.

## 📂 Structure

```
Huffman_coding/
├── huffman_core.py          # shared Huffman logic (build code, encode, decode, file I/O)
├── Huffman_coding.py        # grayscale compression
├── Huffman_coding_RGB.py    # colour compression (one code per channel)
├── input_image.jpg          # <- your input image (add this)
├── results/                 # generated outputs
├── README.md
└── requirements.txt
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Place your image in this folder as `input_image.jpg`, then run:

```bash
python Huffman_coding.py          # grayscale
python Huffman_coding_RGB.py      # RGB
```

Use another image: `python Huffman_coding.py path/to/photo.png`

## 🔍 How it works

1. Count how often each pixel value occurs.
2. Repeatedly merge the two least frequent groups; each merge makes every
   symbol in those groups one bit longer → gives the code length per symbol.
3. Assign **canonical codes** (sorted by length, then value).
4. Replace every pixel by its code and pack the bits into bytes → `.bin` file.
5. Read the `.bin` back, decode, and verify the image is **identical** (lossless).

## 📊 Output (in `results/`)

| File | Description |
|------|-------------|
| `compressed_gray.bin` / `compressed_rgb.bin` | Compressed data |
| `reconstructed_*.png` | Image decoded from the `.bin` file |
| `huffman_*_result.png` | Original vs reconstructed + histogram |
| `code_lengths_gray.png` | Code length per gray level |
| `stats_*.txt` | Compression ratio, entropy, avg code length |

## 📝 Notes

- Average code length is always within 1 bit of the entropy.
- Decoding is pure Python, so very large images (> ~10 MP) may take a while.
- Natural photos compress modestly (8 bpp → ~6–7.5 bpp); this is normal for
  Huffman on raw pixel values.
