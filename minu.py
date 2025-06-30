import os
import cv2
import numpy as np
import torch
from torchvision import transforms
from skimage.morphology import skeletonize
from skimage.filters import sobel
from skimage.util import img_as_ubyte
import matplotlib.pyplot as plt

# Define paths
CONTACT_PATH = "fingerprints/contact-based_fingerprints/first_session"
CONTACTLESS_PATH = "fingerprints/contactless_2d_fingerprint_images/first_session"
OUTPUT_PATH = "f_processed_output/"
os.makedirs(OUTPUT_PATH, exist_ok=True)


# Function to extract minutiae points
def extract_minutiae(image_path, output_folder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Step 1: Binarization
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Step 2: Skeletonization
    skeleton = skeletonize(binary // 255)
    skeleton = img_as_ubyte(skeleton)
    cv2.imwrite(os.path.join(output_folder, "skeleton.png"), skeleton)
    
    # Step 3: Minutiae Extraction (Placeholder)
    # This part will need an advanced algorithm to detect ridge endings and bifurcations
    minutiae_points = np.column_stack(np.where(skeleton > 0))  # Simple example extracting all white pixels
    
    # Save the extracted minutiae points
    np.savetxt(os.path.join(output_folder, "minutiae.txt"), minutiae_points, fmt='%d')
    
    return minutiae_points

# Process all preprocessed images
preprocessed_folders = [f for f in os.listdir(OUTPUT_PATH) if os.path.isdir(os.path.join(OUTPUT_PATH, f))]
for folder in preprocessed_folders:
    folder_path = os.path.join(OUTPUT_PATH, folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name, "enhanced.png")
        if os.path.exists(img_path):
            extract_minutiae(img_path, os.path.join(folder_path, img_name))

print("Minutiae extraction completed. Output stored in", OUTPUT_PATH)
