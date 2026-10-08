# Bit-Plane Decomposition

A Digital Image Processing project that separates an image into its eight binary bit planes.

## Features
- Grayscale bit-plane extraction
- Optional RGB channel extraction
- Individual plane images
- Combined montage of the grayscale planes

## Structure
```text
bitplane_decomposition/
├── main.py
├── bitplane_utils.py
├── requirements.txt
├── README.md
└── output/
```

## Install
```bash
pip install -r requirements.txt
```

## Run
```bash
python main.py input.jpg
```

For RGB planes:
```bash
python main.py input.jpg --rgb
```

Generated files are stored in `output/`.

## Principle
An 8-bit pixel can be represented as:

b7*2^7 + b6*2^6 + ... + b1*2^1 + b0*2^0

Each `bk` forms one bit plane. Higher-order planes generally contain more visually significant intensity information.
