# We decided to use K-means for color quantization 
# We decided not to manually find which pixel is closest to the 6 fixed cube colors (R, W, B, G, Y, O)
# because K-Means finds dominant colors in your specific image, even if those colors don’t match the fixed cube colors closely. So, 
# more natural-looking mosaics, especially for images with subtle or complex tones (e.g., skin, skies, shadows


# so we first tried RGB distance, but realized it's not good b/c that's not how humans perceive color
# so the solution is to use Lab color space (CIELAB) which is designed to match how humans see and perceive colors.
import numpy as np
from sklearn.cluster import KMeans
from PIL import Image
from skimage.color import rgb2lab
from scipy.optimize import linear_sum_assignment
import Image_Preprocessing

img_np = np.array(Image_Preprocessing.resized_image) # get a 3D numpy array with 

print(img_np.shape) # 750 rows (height), 990 columns (width), with 3 RGB values
print(img_np[0, 2]) # red 124, green 155, blue 186, the location is the 3rd pixel from the left

h, w, _ = img_np.shape
pixels = img_np.reshape(-1, 3)  # Flatten to (num_pixels, 3)


# Step 2: Run K-Means clustering
kmeans = KMeans(n_clusters=6, random_state=42)
kmeans.fit(pixels)
cluster_centers = kmeans.cluster_centers_  # RGB values of dominant colors
labels = kmeans.labels_

# Step 3: Replace clustered pixels with nearest Rubik's cube colors
rubik_colors = {
    'W': np.array([255, 255, 255]),
    'Y': np.array([255, 255,   0]),
    'R': np.array([255,   0,   0]),
    'O': np.array([255, 165,   0]),
    'G': np.array([  0, 128,   0]),
    'B': np.array([  0,   0, 255]),
    'Bl': np.array([ 0,   0,   0])
}

rubik_color_names = list(rubik_colors.keys())
rubik_color_values = np.array(list(rubik_colors.values()))

print(rubik_color_names)

# Convert cluster centers and rubik colors to Lab (scikit-image expects float [0,1])
cluster_centers_lab = rgb2lab(cluster_centers[np.newaxis, :] / 255.0)[0]
rubik_lab = rgb2lab(rubik_color_values[np.newaxis, :] / 255.0)[0]

# Create cost matrix using Lab distance
cost_matrix = np.zeros((7, 7))
for i, cluster_lab in enumerate(cluster_centers_lab):
    for j, rubik_lab_color in enumerate(rubik_lab):
        cost_matrix[i, j] = np.linalg.norm(cluster_lab - rubik_lab_color)

# Apply Hungarian algorithm to get best unique assignments
row_ind, col_ind = linear_sum_assignment(cost_matrix)
cluster_to_rubik = {i: rubik_color_names[j] for i, j in zip(row_ind, col_ind)}

# Debug: Print assignments
print("\nCluster centers and assigned Rubik’s colors:")
for i, center_rgb in enumerate(cluster_centers.astype(int)):
    assigned = cluster_to_rubik[i]
    print(f"Cluster {i}: RGB {center_rgb} → Rubik Color: {assigned}")

quantized_pixels = np.array([
    rubik_colors[cluster_to_rubik[label]] for label in labels
])
quantized_img = quantized_pixels.reshape(h, w, 3).astype(np.uint8)
quantized_image = Image.fromarray(quantized_img)
quantized_image.show()



# --- Cube mosaic extraction ---
cube_face_size = 30  # Each cube is 30x30 pixels
facelet_size = cube_face_size // 3  # Each facelet is 10x10

cubes_wide = w // cube_face_size
cubes_high = h // cube_face_size

cube_patterns = []

# Map RGB back to Rubik color name (reverse dict)
rgb_to_rubik = {tuple(v): k for k, v in rubik_colors.items()}

for row in range(cubes_high):
    for col in range(cubes_wide):
        block = quantized_img[
            row * cube_face_size:(row + 1) * cube_face_size,
            col * cube_face_size:(col + 1) * cube_face_size
        ]

        cube_face = []

        for i in range(3):  # rows of facelets
            facelet_row = []
            for j in range(3):  # columns of facelets
                subregion = block[
                    i * facelet_size:(i + 1) * facelet_size,
                    j * facelet_size:(j + 1) * facelet_size
                ]
                avg_rgb = np.mean(subregion.reshape(-1, 3), axis=0)
                distances = [np.linalg.norm(avg_rgb - rubik_color_values[k]) for k in range(len(rubik_color_values))]
                nearest_idx = np.argmin(distances)
                rubik_color = rubik_color_names[nearest_idx]
                facelet_row.append(rubik_color)
            cube_face.append(facelet_row)

        cube_patterns.append(cube_face)

# --- Optional: Display first few cube patterns ---
print("\nSample cube face patterns:")
for i, pattern in enumerate(cube_patterns[:5]):
    for row in pattern:
        print(row)
    print("---------")



import matplotlib.pyplot as plt

# Show the full quantized Rubik's cube mosaic image
plt.figure(figsize=(10, 8))
plt.imshow(quantized_img)
plt.title("Rubik's Cube Mosaic (Quantized)")
plt.axis('off')
plt.tight_layout()
plt.show()
