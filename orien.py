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

# Function to compute fingerprint orientation field
def compute_orientation_field(image_path, output_folder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Compute gradients using Sobel filter
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=5)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=5)
    orientation_field = np.arctan2(sobely, sobelx)
    
    # Normalize and save the orientation field
    orientation_field = (orientation_field - np.min(orientation_field)) / (np.max(orientation_field) - np.min(orientation_field)) * 255
    orientation_field = orientation_field.astype(np.uint8)
    cv2.imwrite(os.path.join(output_folder, "orientation_field.png"), orientation_field)
    
    return orientation_field

# Process all preprocessed images
preprocessed_folders = [f for f in os.listdir(OUTPUT_PATH) if os.path.isdir(os.path.join(OUTPUT_PATH, f))]
for folder in preprocessed_folders:
    folder_path = os.path.join(OUTPUT_PATH, folder)
    for img_name in os.listdir(folder_path):
        img_path = os.path.join(folder_path, img_name, "enhanced.png")
        if os.path.exists(img_path):
            compute_orientation_field(img_path, os.path.join(folder_path, img_name))

print("Orientation field computation completed. Output stored in", OUTPUT_PATH)
