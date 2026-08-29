import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. GLOBAL HISTOGRAM EQUALIZATION
# ============================================================
def global_histogram_equalization(image):
    """
    Applies histogram equalization to the complete image.
    """
    return cv2.equalizeHist(image)


# ============================================================
# 2. LOCAL HISTOGRAM EQUALIZATION
# ============================================================
def local_histogram_equalization(image, window_size=31):
    """
    Local histogram equalization.

    A small window is moved over the image.
    The center pixel is equalized using the histogram
    of pixels inside that local window.
    """

    pad = window_size // 2

    # Pad image to handle boundaries
    padded = cv2.copyMakeBorder(
        image,
        pad, pad, pad, pad,
        cv2.BORDER_REFLECT
    )

    output = np.zeros_like(image)

    rows, cols = image.shape

    for i in range(rows):
        for j in range(cols):

            # Extract local window
            window = padded[
                i:i + window_size,
                j:j + window_size
            ]

            # Histogram
            hist = np.bincount(
                window.flatten(),
                minlength=256
            )

            # CDF
            cdf = hist.cumsum()

            # Ignore zero values in CDF
            cdf_min = cdf[cdf > 0][0]

            # Number of pixels
            n = window.size

            # Equalization mapping
            if n > cdf_min:
                equalized_value = (
                    (cdf[image[i, j]] - cdf_min)
                    * 255
                    / (n - cdf_min)
                )
            else:
                equalized_value = image[i, j]

            output[i, j] = np.clip(
                equalized_value,
                0,
                255
            )

    return output.astype(np.uint8)


# ============================================================
# 3. ADAPTIVE HISTOGRAM EQUALIZATION (AHE)
# ============================================================
def adaptive_histogram_equalization(image, tile_size=32):
    """
    Basic Adaptive Histogram Equalization.

    The image is divided into small regions (tiles),
    and histogram equalization is applied independently
    to each tile.
    """

    output = np.zeros_like(image)

    rows, cols = image.shape

    for y in range(0, rows, tile_size):
        for x in range(0, cols, tile_size):

            # Get tile
            tile = image[
                y:min(y + tile_size, rows),
                x:min(x + tile_size, cols)
            ]

            # Equalize tile
            equalized_tile = cv2.equalizeHist(tile)

            # Put result back
            output[
                y:min(y + tile_size, rows),
                x:min(x + tile_size, cols)
            ] = equalized_tile

    return output


# ============================================================
# 4. CLAHE
# Contrast Limited Adaptive Histogram Equalization
# ============================================================
def clahe_equalization(image, clip_limit=2.0, tile_grid_size=(8, 8)):
    """
    CLAHE prevents excessive contrast enhancement
    by clipping the histogram.
    """

    clahe = cv2.createCLAHE(
        clipLimit=clip_limit,
        tileGridSize=tile_grid_size
    )

    return clahe.apply(image)


# ============================================================
# 5. VLAHE
# Variable Local Area Histogram Equalization
# ============================================================
def vlahe_equalization(image, min_window=9, max_window=31):
    """
    Simple VLAHE implementation.

    The local window size changes according to
    local image variance.

    Low variance  -> smaller window
    High variance -> larger window
    """

    rows, cols = image.shape

    output = np.zeros_like(image)

    pad = max_window // 2

    padded = cv2.copyMakeBorder(
        image,
        pad, pad, pad, pad,
        cv2.BORDER_REFLECT
    )

    for i in range(rows):
        for j in range(cols):

            # Start with maximum window
            window = padded[
                i:i + max_window,
                j:j + max_window
            ]

            # Calculate local variance
            variance = np.var(window)

            # Normalize variance approximately to 0-1
            normalized_variance = min(
                variance / 5000.0,
                1.0
            )

            # Select variable window size
            window_size = int(
                max_window
                - normalized_variance
                * (max_window - min_window)
            )

            # Make window odd
            if window_size % 2 == 0:
                window_size += 1

            window_size = max(
                min_window,
                min(window_size, max_window)
            )

            half = window_size // 2

            # Coordinates in padded image
            center_i = i + pad
            center_j = j + pad

            local_window = padded[
                center_i - half:center_i + half + 1,
                center_j - half:center_j + half + 1
            ]

            # Histogram
            hist = np.bincount(
                local_window.flatten(),
                minlength=256
            )

            # CDF
            cdf = hist.cumsum()

            cdf_nonzero = cdf[cdf > 0]

            if len(cdf_nonzero) == 0:
                output[i, j] = image[i, j]
                continue

            cdf_min = cdf_nonzero[0]

            total_pixels = local_window.size

            current_pixel = image[i, j]

            if total_pixels != cdf_min:

                value = (
                    (cdf[current_pixel] - cdf_min)
                    * 255
                    / (total_pixels - cdf_min)
                )

            else:
                value = current_pixel

            output[i, j] = np.clip(value, 0, 255)

    return output.astype(np.uint8)


# ============================================================
# HISTOGRAM DISPLAY FUNCTION
# ============================================================
def show_histogram(image, title):
    plt.hist(
        image.ravel(),
        bins=256,
        range=(0, 256),
        color='black'
    )

    plt.title(title)
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.xlim([0, 256])


# ============================================================
# MAIN PROGRAM
# ============================================================

# Read image in grayscale
image = cv2.imread(
    "input.jpg",
    cv2.IMREAD_GRAYSCALE
)

if image is None:
    print("Error: Could not load image.")
    print("Make sure 'input.jpg' exists.")
    exit()


# Apply different histogram equalization methods

global_result = global_histogram_equalization(image)

# Local histogram equalization
# Small window = stronger local enhancement
local_result = local_histogram_equalization(
    image,
    window_size=15
)

# Adaptive histogram equalization
ahe_result = adaptive_histogram_equalization(
    image,
    tile_size=32
)

# CLAHE
clahe_result = clahe_equalization(
    image,
    clip_limit=2.0,
    tile_grid_size=(8, 8)
)

# VLAHE
vlahe_result = vlahe_equalization(
    image,
    min_window=9,
    max_window=31
)


# ============================================================
# DISPLAY RESULTS
# ============================================================

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
plt.imshow(image, cmap='gray')
plt.title("Original Image")
plt.axis("off")

plt.subplot(2, 3, 2)
plt.imshow(global_result, cmap='gray')
plt.title("Global Histogram Equalization")
plt.axis("off")

plt.subplot(2, 3, 3)
plt.imshow(local_result, cmap='gray')
plt.title("Local Histogram Equalization")
plt.axis("off")

plt.subplot(2, 3, 4)
plt.imshow(ahe_result, cmap='gray')
plt.title("Adaptive Histogram Equalization")
plt.axis("off")

plt.subplot(2, 3, 5)
plt.imshow(clahe_result, cmap='gray')
plt.title("CLAHE")
plt.axis("off")

plt.subplot(2, 3, 6)
plt.imshow(vlahe_result, cmap='gray')
plt.title("VLAHE")
plt.axis("off")

plt.tight_layout()
plt.show()


# ============================================================
# DISPLAY HISTOGRAMS
# ============================================================

plt.figure(figsize=(15, 10))

plt.subplot(2, 3, 1)
show_histogram(image, "Original Histogram")

plt.subplot(2, 3, 2)
show_histogram(global_result, "Global HE")

plt.subplot(2, 3, 3)
show_histogram(local_result, "Local HE")

plt.subplot(2, 3, 4)
show_histogram(ahe_result, "Adaptive HE")

plt.subplot(2, 3, 5)
show_histogram(clahe_result, "CLAHE")

plt.subplot(2, 3, 6)
show_histogram(vlahe_result, "VLAHE")

plt.tight_layout()
plt.show()
