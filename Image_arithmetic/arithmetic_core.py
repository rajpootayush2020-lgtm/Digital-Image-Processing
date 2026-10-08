"""
arithmetic_core.py
Shared helpers for pixel-wise image arithmetic. Every function works on
uint8 arrays of any shape (grayscale H x W or colour H x W x 3), so the
grayscale and RGB scripts can share them.
"""

import numpy as np
from PIL import Image


def load_pair(path1, path2, mode):
    """Load two images in `mode` ('L' or 'RGB'); resize the 2nd to match the 1st."""
    a = Image.open(path1).convert(mode)
    b = Image.open(path2).convert(mode)
    if b.size != a.size:
        b = b.resize(a.size, Image.LANCZOS)
    return np.array(a, dtype=np.uint8), np.array(b, dtype=np.uint8)


def clip8(x):
    """Saturate to 0..255 and return uint8."""
    return np.clip(x, 0, 255).astype(np.uint8)


def normalize8(x):
    """Min-max stretch any array into 0..255 (uint8)."""
    x = x.astype(np.float64)
    lo, hi = x.min(), x.max()
    if hi == lo:
        return np.zeros(x.shape, dtype=np.uint8)
    return ((x - lo) / (hi - lo) * 255).astype(np.uint8)


# ---------------------------- two-image operations ---------------------------
def add_saturated(a, b):
    return clip8(a.astype(np.int32) + b.astype(np.int32))


def add_wraparound(a, b):
    """Plain uint8 overflow: 200 + 100 -> 44 (shown only for comparison)."""
    return (a + b).astype(np.uint8)


def subtract(a, b):
    return clip8(a.astype(np.int32) - b.astype(np.int32))


def abs_difference(a, b):
    return np.abs(a.astype(np.int32) - b.astype(np.int32)).astype(np.uint8)


def multiply(a, b):
    """(a * b) / 255 keeps the result in 0..255."""
    return clip8(a.astype(np.float64) * b.astype(np.float64) / 255.0)


def divide(a, b):
    """a / (b + 1), stretched to 0..255 (avoids division by zero).
    The top 1 % of ratios is clipped first so a few extreme pixels
    (where B is almost 0) do not make the whole result look dark."""
    ratio = a.astype(np.float64) / (b.astype(np.float64) + 1.0)
    ratio = np.minimum(ratio, np.percentile(ratio, 99))
    return normalize8(ratio)


def average(a, b):
    return clip8((a.astype(np.float64) + b.astype(np.float64)) / 2.0)


def blend(a, b, alpha=0.7):
    """alpha * A + (1 - alpha) * B"""
    return clip8(alpha * a.astype(np.float64) + (1 - alpha) * b.astype(np.float64))


# --------------------------- single-image operations -------------------------
def brightness(a, value=60):
    return clip8(a.astype(np.int32) + value)


def contrast(a, factor=1.5):
    return clip8(a.astype(np.float64) * factor)


def negative(a):
    return (255 - a).astype(np.uint8)


# ------------------------------ convenience list -----------------------------
def all_operations(a, b, alpha=0.7):
    """Return an ordered dict {file_stem: (title, result_array)}."""
    return {
        "01_addition":        ("A + B (saturated)", add_saturated(a, b)),
        "02_addition_wrap":   ("A + B (wrap-around)", add_wraparound(a, b)),
        "03_subtraction":     ("A - B", subtract(a, b)),
        "04_abs_difference":  ("|A - B|", abs_difference(a, b)),
        "05_multiplication":  ("A x B / 255", multiply(a, b)),
        "06_division":        ("A / (B + 1), normalized", divide(a, b)),
        "07_average":         ("(A + B) / 2", average(a, b)),
        "08_blend":           (f"Blend {alpha:.1f}A + {1 - alpha:.1f}B", blend(a, b, alpha)),
        "09_brightness":      ("A + 60 (brighter)", brightness(a, 60)),
        "10_contrast":        ("A x 1.5 (contrast)", contrast(a, 1.5)),
        "11_negative":        ("255 - A (negative)", negative(a)),
    }
