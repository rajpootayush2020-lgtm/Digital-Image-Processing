"""
huffman_core.py
Shared Huffman helpers (canonical Huffman coding) used by the grayscale
and RGB scripts.

Idea: instead of storing the whole tree, we only compute the *length* of every
symbol's code. Codes are then rebuilt in "canonical" order, so the decoder
needs just 256 length values (one byte each) to recreate the same codes.
"""

import heapq
import itertools
import struct
from collections import Counter

import numpy as np

MAGIC = b"HUFF"


# ----------------------------------------------------------------------------
# Building the code
# ----------------------------------------------------------------------------
def code_lengths(freq):
    """Return {symbol: code_length} from {symbol: count} using the Huffman merge."""
    if len(freq) == 1:                       # image with a single pixel value
        return {s: 1 for s in freq}

    tie = itertools.count()                  # keeps heap ordering stable
    heap = [(count, next(tie), [sym]) for sym, count in freq.items()]
    heapq.heapify(heap)
    depth = {sym: 0 for sym in freq}

    while len(heap) > 1:
        c1, _, group1 = heapq.heappop(heap)
        c2, _, group2 = heapq.heappop(heap)
        for sym in group1 + group2:          # every merge pushes symbols 1 level deeper
            depth[sym] += 1
        heapq.heappush(heap, (c1 + c2, next(tie), group1 + group2))
    return depth


def canonical_codes(lengths):
    """Return {symbol: 'bitstring'} built in canonical order from the lengths."""
    codes = {}
    code, prev_len = 0, 0
    for sym, length in sorted(lengths.items(), key=lambda kv: (kv[1], kv[0])):
        code <<= (length - prev_len)
        codes[sym] = format(code, f"0{length}b")
        code += 1
        prev_len = length
    return codes


# ----------------------------------------------------------------------------
# Encoding / decoding one channel (flat uint8 array)
# ----------------------------------------------------------------------------
def encode_channel(flat):
    """Return (lengths_dict, payload_bytes, number_of_bits)."""
    freq = Counter(flat.tolist())
    lengths = code_lengths(freq)
    codes = canonical_codes(lengths)

    lookup = [""] * 256
    for sym, bits in codes.items():
        lookup[sym] = bits
    bitstring = "".join(map(lookup.__getitem__, flat.tolist()))

    nbits = len(bitstring)
    bit_array = np.frombuffer(bitstring.encode("ascii"), dtype=np.uint8) - 48
    payload = np.packbits(bit_array).tobytes()
    return lengths, payload, nbits


def decode_channel(lengths, payload, nbits, count):
    """Rebuild `count` pixel values from the payload."""
    codes = canonical_codes(lengths)
    table = {(len(b), int(b, 2)): sym for sym, b in codes.items()}

    bits = np.unpackbits(np.frombuffer(payload, dtype=np.uint8))[:nbits].tolist()
    out = np.empty(count, dtype=np.uint8)
    code, length, idx = 0, 0, 0
    for bit in bits:
        code = (code << 1) | bit
        length += 1
        sym = table.get((length, code))
        if sym is not None:
            out[idx] = sym
            idx += 1
            code, length = 0, 0
    return out


# ----------------------------------------------------------------------------
# Container file (.bin)
# ----------------------------------------------------------------------------
def write_bin(path, height, width, channels):
    """channels = list of (lengths_dict, payload_bytes, nbits)."""
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<BII", len(channels), height, width))
        for lengths, payload, nbits in channels:
            table = bytes(lengths.get(s, 0) for s in range(256))
            f.write(table)
            f.write(struct.pack("<QQ", nbits, len(payload)))
            f.write(payload)


def read_bin(path):
    with open(path, "rb") as f:
        assert f.read(4) == MAGIC, "Not a valid HUFF file"
        n_ch, height, width = struct.unpack("<BII", f.read(9))
        channels = []
        for _ in range(n_ch):
            table = f.read(256)
            nbits, plen = struct.unpack("<QQ", f.read(16))
            payload = f.read(plen)
            lengths = {s: l for s, l in enumerate(table) if l > 0}
            channels.append((lengths, payload, nbits))
    return height, width, channels


# ----------------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------------
def entropy(flat):
    counts = np.bincount(flat, minlength=256).astype(float)
    p = counts[counts > 0] / counts.sum()
    return float(-(p * np.log2(p)).sum())


def average_length(flat, lengths):
    counts = np.bincount(flat, minlength=256)
    total = counts.sum()
    return float(sum(counts[s] * l for s, l in lengths.items()) / total)
