"""
shannon_fano_core.py
Shared Shannon-Fano helpers used by the grayscale and RGB scripts.

Shannon-Fano is a top-down method:
  1. Sort the symbols by frequency (highest first).
  2. Split the list into two parts whose total frequencies are as equal as possible.
  3. Give '0' to the upper part and '1' to the lower part.
  4. Repeat on each part until every part holds a single symbol.
(Huffman, in contrast, builds its tree bottom-up.)
"""

import struct
from collections import Counter

import numpy as np

MAGIC = b"SFAN"


# ----------------------------------------------------------------------------
# Building the code
# ----------------------------------------------------------------------------
def best_split(items):
    """items = [(symbol, count), ...] sorted by count desc. Return split index."""
    total = sum(c for _, c in items)
    running, best_idx, best_diff = 0, 1, None
    for i in range(len(items) - 1):
        running += items[i][1]
        diff = abs(total - 2 * running)          # |left - right|
        if best_diff is None or diff < best_diff:
            best_diff, best_idx = diff, i + 1
    return best_idx


def shannon_fano_codes(freq):
    """Return {symbol: 'bitstring'} from {symbol: count}."""
    items = sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))
    if len(items) == 1:
        return {items[0][0]: "0"}

    codes = {sym: "" for sym, _ in items}
    stack = [items]
    while stack:
        group = stack.pop()
        if len(group) == 1:
            continue
        cut = best_split(group)
        upper, lower = group[:cut], group[cut:]
        for sym, _ in upper:
            codes[sym] += "0"
        for sym, _ in lower:
            codes[sym] += "1"
        stack.append(upper)
        stack.append(lower)
    return codes


# ----------------------------------------------------------------------------
# Encoding / decoding one channel (flat uint8 array)
# ----------------------------------------------------------------------------
def encode_channel(flat):
    """Return (codes_dict, payload_bytes, number_of_bits)."""
    freq = Counter(flat.tolist())
    codes = shannon_fano_codes(freq)

    lookup = [""] * 256
    for sym, bits in codes.items():
        lookup[sym] = bits
    bitstring = "".join(map(lookup.__getitem__, flat.tolist()))

    nbits = len(bitstring)
    bit_array = np.frombuffer(bitstring.encode("ascii"), dtype=np.uint8) - 48
    payload = np.packbits(bit_array).tobytes()
    return codes, payload, nbits


def decode_channel(codes, payload, nbits, count):
    """Rebuild `count` pixel values from the payload."""
    table = {bits: sym for sym, bits in codes.items()}
    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))[:nbits].tolist()

    out = np.empty(count, dtype=np.uint8)
    current, idx = "", 0
    for b in bits:
        current += "1" if b else "0"
        sym = table.get(current)
        if sym is not None:
            out[idx] = sym
            idx += 1
            current = ""
    return out


# ----------------------------------------------------------------------------
# Container file (.bin)  -  stores the real Shannon-Fano code table
# ----------------------------------------------------------------------------
def write_bin(path, height, width, channels):
    """channels = list of (codes_dict, payload_bytes, nbits)."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<BII", len(channels), height, width))
        for codes, payload, nbits in channels:
            f.write(struct.pack("<H", len(codes)))
            for sym, bits in sorted(codes.items()):
                nbytes = (len(bits) + 7) // 8
                f.write(struct.pack("<BB", sym, len(bits)))
                f.write(int(bits, 2).to_bytes(nbytes, "big"))
            f.write(struct.pack("<QQ", nbits, len(payload)))
            f.write(payload)


def read_bin(path):
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC, "Not a valid SFAN file"
        n_ch, height, width = struct.unpack("<BII", f.read(9))
        channels = []
        for _ in range(n_ch):
            (n_sym,) = struct.unpack("<H", f.read(2))
            codes = {}
            for _ in range(n_sym):
                sym, length = struct.unpack("<BB", f.read(2))
                nbytes = (length + 7) // 8
                value = int.from_bytes(f.read(nbytes), "big")
                codes[sym] = format(value, f"0{length}b")
            nbits, plen = struct.unpack("<QQ", f.read(16))
            channels.append((codes, f.read(plen), nbits))
    return height, width, channels


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def entropy(flat):
    counts = np.bincount(flat, minlength=256).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def average_length(flat, codes):
    counts = np.bincount(flat, minlength=256)
    return float(sum(counts[s] * len(b) for s, b in codes.items()) / counts.sum())
