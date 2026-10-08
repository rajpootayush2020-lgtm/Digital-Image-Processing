# 🗜️ Shannon-Fano Coding (Lossless Image Compression)

Lossless image compression using **Shannon-Fano coding**, a top-down entropy
coding method. The full code table is stored inside the compressed `.bin` file,
so the decoder needs nothing else.

## 📂 Structure

```
Shannon_fano_coding/
├── shannon_fano_core.py          # shared logic (build code, encode, decode, file I/O)
├── Shannon_fano_coding.py        # grayscale compression
├── Shannon_fano_coding_RGB.py    # colour compression (one code per channel)
├── input_image.jpg               # <- your input image (add this)
├── results/                      # generated outputs
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
python Shannon_fano_coding.py         # grayscale
python Shannon_fano_coding_RGB.py     # RGB
```

Use another image: `python Shannon_fano_coding.py path/to/photo.png`

## 🔍 How it works

1. Count how often each pixel value occurs and sort symbols by frequency (high → low).
2. Split the list into two parts with totals as close as possible.
3. Upper part gets bit `0`, lower part gets bit `1`.
4. Repeat on each part until every part has one symbol.
5. Replace every pixel with its code, pack bits into bytes, and save the `.bin` file.
6. Read the `.bin` back, decode, and verify the image is **identical** (lossless).

## 📊 Output (in `results/`)

| File | Description |
|------|-------------|
| `compressed_gray.bin` / `compressed_rgb.bin` | Compressed data + code table |
| `reconstructed_*.png` | Image decoded from the `.bin` file |
| `shannon_fano_*_result.png` | Original vs reconstructed + histogram |
| `code_lengths_gray.png` | Code length per gray level |
| `code_table_gray.txt` | Pixel value, count and its code |
| `stats_*.txt` | Compression ratio, entropy, average length, efficiency |

## 📝 Notes

- Shannon-Fano is not always optimal; Huffman coding is never worse on average.
  Comparing both on the same image makes a good analysis.
- Decoding is pure Python, so very large images (> ~10 MP) may take a while.
