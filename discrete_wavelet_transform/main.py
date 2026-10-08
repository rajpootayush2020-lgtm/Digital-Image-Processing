from pathlib import Path
from dwt_utils import load_grayscale, haar_dwt, save_subbands, make_montage

def main():
    base = Path(__file__).parent
    image = base / "input.png"
    output = base / "output"
    output.mkdir(exist_ok=True)
    img = load_grayscale(image)
    LL, LH, HL, HH = haar_dwt(img)
    save_subbands(LL, LH, HL, HH, output)
    make_montage(LL, LH, HL, HH, output / "dwt_subbands.png")
    print("Discrete Wavelet Transform completed.")

if __name__ == "__main__":
    main()
