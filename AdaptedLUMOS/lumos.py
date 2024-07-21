import cupy as cp
from cuml.cluster import KMeans as cuKMeans
from skimage.filters import median
from skimage.morphology import disk
from PIL import Image, ImageOps
import numpy as np
import time

def preprocess_image(image):
    # Normalize each channel separately
    mean = cp.mean(image)
    std = cp.std(image)
    scaled_image = (image - mean) / std
    return scaled_image

def apply_median_filter(image, size):
    # Apply median filter to the image
    filtered_image = median(cp.asnumpy(image), disk(size))
    return cp.array(filtered_image)

def lumos(image, num_clusters, max_iter=100, num_replicates=10):
    # Preprocess image
    scaled_image = preprocess_image(image)
    
    # Apply median filter
    filtered_image = apply_median_filter(scaled_image, size=3)
    
    # Reshape image for k-means
    pixels = filtered_image.reshape(-1, 1)
    
    # Initialize cuML k-means with k-means++
    kmeans = cuKMeans(n_clusters=num_clusters, init='scalable-k-means++', max_iter=max_iter, n_init=num_replicates)
    
    # Fit k-means on the GPU

    start = time.time()
    kmeans.fit(pixels)
    end = time.time()
    # Assign clusters
    labels = kmeans.predict(pixels)
    labels_image = labels.reshape(image.shape[:2])
    
    print("prediction time:\t", end - start)
    return labels_image

def apply_color_map(labels_image, color_map):
    # Create an RGB image based on the cluster labels
    height, width = labels_image.shape
    color_image = cp.zeros((height, width, 3), dtype=cp.uint8)
    
    for label, color in color_map.items():
        mask = (labels_image == label)
        for i in range(3):  # Apply the RGB channels
            color_image[:, :, i][mask] = color[i]
    
    return color_image

# Define a color map for the clusters
# This currently uses White for the secondary signal and Black for the primary signal and surrounding environment
# It may be necessary to change which is White.
color_map = {
    0: (0, 0, 0),  # Black
    1: (0, 0, 0),  # Black
    2: (255, 255, 255) # White
}

# Example usage:
image_path = "put the path to the input image her"

print("opening image")
image = Image.open(image_path)

print("ensuring in greyscale")
image = ImageOps.grayscale(image)

print("numpy array")
image = np.array(image)

print("cupy array")
image = cp.array(image)

# Number of clusters (fluorophores + background)
num_clusters = 3

# Run LUMoS
print("Running LUMoS")
labels_image = lumos(image, num_clusters)

# Apply color map to the labels
print("Applying color map")
color_image = apply_color_map(labels_image, color_map)

# Convert CuPy array back to NumPy array for saving
print("Converting to NumPy array")
color_image_np = cp.asnumpy(color_image)

# Convert NumPy array to PIL Image
print("Converting to PIL Image")
color_image_pil = Image.fromarray(color_image_np)

# Save the image
output_path = "put the desired output path here"
print("Saving the image")
color_image_pil.save(output_path)

print(f"Image saved to {output_path}")
