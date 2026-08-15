import cv2
import numpy as np

# Read input image
image = cv2.imread("original.jpg")

if image is None:
    print("Error: original.jpg not found!")
    exit()

# Convert to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Split BGR channels
blue, green, red = cv2.split(image)

# Create Red channel image
red_image = cv2.merge([
    np.zeros_like(blue),
    np.zeros_like(green),
    red
])

# Create Green channel image
green_image = cv2.merge([
    np.zeros_like(blue),
    green,
    np.zeros_like(red)
])

# Create Blue channel image
blue_image = cv2.merge([
    blue,
    np.zeros_like(green),
    np.zeros_like(red)
])

# Display images
cv2.imshow("Original", image)
cv2.imshow("Grayscale", gray)
cv2.imshow("Red", red_image)
cv2.imshow("Green", green_image)
cv2.imshow("Blue", blue_image)

# Save output images
cv2.imwrite("grayscale.jpg", gray)
cv2.imwrite("red.jpg", red_image)
cv2.imwrite("green.jpg", green_image)
cv2.imwrite("blue.jpg", blue_image)

# Wait for key press
cv2.waitKey(0)

# Close windows
cv2.destroyAllWindows()