from pathlib import Path
import numpy as np
from PIL import Image


def read_gray(path):
    return np.asarray(Image.open(path).convert('L'), dtype=np.float32)


def convolve(image, kernel):
    kh, kw = kernel.shape
    ph, pw = kh // 2, kw // 2
    padded = np.pad(image, ((ph, ph), (pw, pw)), mode='edge')
    result = np.zeros_like(image, dtype=np.float32)

    for r in range(image.shape[0]):
        for c in range(image.shape[1]):
            result[r, c] = np.sum(padded[r:r+kh, c:c+kw] * kernel)
    return result


def scale_to_uint8(data):
    data = np.abs(data)
    lo, hi = data.min(), data.max()
    if hi == lo:
        return np.zeros_like(data, dtype=np.uint8)
    return np.clip((data - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


def edge_map(magnitude, threshold=70):
    return np.where(magnitude >= threshold, 255, 0).astype(np.uint8)


def save_image(array, path):
    Image.fromarray(array.astype(np.uint8), mode='L').save(path)


def main():
    root = Path(__file__).parent
    input_file = root / 'input.png'
    out = root / 'output'
    out.mkdir(exist_ok=True)

    image = read_gray(input_file)

    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float32)

    prewitt_x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float32)
    prewitt_y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float32)

    laplacian = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=np.float32)

    sobel_gx = convolve(image, sobel_x)
    sobel_gy = convolve(image, sobel_y)
    sobel_mag = np.sqrt(sobel_gx ** 2 + sobel_gy ** 2)

    prewitt_gx = convolve(image, prewitt_x)
    prewitt_gy = convolve(image, prewitt_y)
    prewitt_mag = np.sqrt(prewitt_gx ** 2 + prewitt_gy ** 2)

    lap = convolve(image, laplacian)

    save_image(scale_to_uint8(sobel_gx), out / 'sobel_x.png')
    save_image(scale_to_uint8(sobel_gy), out / 'sobel_y.png')
    save_image(edge_map(sobel_mag), out / 'sobel_edges.png')
    save_image(scale_to_uint8(prewitt_gx), out / 'prewitt_x.png')
    save_image(scale_to_uint8(prewitt_gy), out / 'prewitt_y.png')
    save_image(edge_map(prewitt_mag), out / 'prewitt_edges.png')
    save_image(scale_to_uint8(lap), out / 'laplacian_edges.png')

    print('Edge detection completed.')
    print('Sobel, Prewitt and Laplacian results are in the output folder.')


if __name__ == '__main__':
    main()
