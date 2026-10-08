from pathlib import Path
from bitplane_utils import load_image, save_gray_planes, make_montage


def main():
    base = Path(__file__).parent

    image = base / "input.png"
    output = base / "output"

    output.mkdir(exist_ok=True)

    img = load_image(image)

    planes = save_gray_planes(img, output / "grayscale")

    make_montage(
        planes,
        output / "grayscale_bitplanes.png"
    )

    print("Bit-plane extraction completed.")


if __name__ == "__main__":
    main()