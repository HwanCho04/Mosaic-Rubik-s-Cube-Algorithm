import numpy as np
import Color_Quantization_Clustering
from PIL import Image

# brute-force approach
from skimage.color import rgb2lab

# Step 1: Convert image to Lab (scale to 0-1 first)
img_lab = rgb2lab(Color_Quantization_Clustering.img_np / 255.0)

# Step 2: Convert Rubik palette to Lab
rubik_lab = rgb2lab(Color_Quantization_Clustering.rubik_color_values[np.newaxis, :] / 255.0)[0]

# Step 3: Brute-force assign each pixel to nearest Rubik color
brute_pixels = np.zeros_like(Color_Quantization_Clustering.img_np)

for i in range(Color_Quantization_Clustering.h):
    for j in range(Color_Quantization_Clustering.w):
        pixel_lab = img_lab[i, j]
        distances = [np.linalg.norm(pixel_lab - rc) for rc in rubik_lab]
        nearest_idx = np.argmin(distances)
        brute_pixels[i, j] = Color_Quantization_Clustering.rubik_color_values[nearest_idx]

# Step 4: Convert to image and show
brute_image = Image.fromarray(brute_pixels.astype(np.uint8))
brute_image.show()