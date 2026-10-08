# ➕ Image Arithmetic

Pixel-wise arithmetic between **two images** (A and B) plus a few single-image
operations, for both grayscale and RGB images.

## 📂 Structure

```
Image_arithmetic/
├── arithmetic_core.py        # all arithmetic functions (shared)
├── Image_arithmetic.py       # grayscale version
├── Image_arithmetic_RGB.py   # colour version
├── input_image1.jpg          # <- image A (add this)
├── input_image2.jpg          # <- image B (add this)
├── results/                  # generated outputs
├── README.md
└── requirements.txt
```

## ⚙️ Installation

```bash
pip install -r requirements.txt
```

## ▶️ Usage

Place two images in this folder as `input_image1.jpg` and `input_image2.jpg`, then:

```bash
python Image_arithmetic.py         # grayscale
python Image_arithmetic_RGB.py     # RGB
```

Custom paths: `python Image_arithmetic.py a.png b.png`
(If B has a different size, it is resized automatically to match A.)

## 🧮 Operations

| Operation | Formula | Typical use |
|-----------|---------|-------------|
| Addition (saturated) | `min(A + B, 255)` | Combine / overlay images |
| Addition (wrap-around) | `(A + B) mod 256` | Shows the overflow problem |
| Subtraction | `max(A - B, 0)` | Remove background |
| Absolute difference | `\|A - B\|` | Change / motion detection |
| Multiplication | `A × B / 255` | Masking, shading |
| Division | `A / (B + 1)`, normalized | Illumination correction |
| Average | `(A + B) / 2` | Noise reduction, mixing |
| Blend | `α·A + (1-α)·B` | Cross-fade (α = 0.7) |
| Brightness | `A + 60` | Brighten |
| Contrast | `A × 1.5` | Increase contrast |
| Negative | `255 - A` | Invert |

## 📊 Output (in `results/`)

- One PNG per operation (`01_addition.png`, `04_abs_difference.png`, ...)
- `arithmetic_gray_overview.png` / `arithmetic_rgb_overview.png`: all results in one figure
- `stats.txt` / `stats_rgb.txt`: mean, min and max of every result

## 📝 Notes

- Results are clipped to 0–255 (saturation) except the wrap-around example.
- Division adds 1 to B to avoid division by zero, clips the top 1 % of ratios, then stretches the result to 0–255.
