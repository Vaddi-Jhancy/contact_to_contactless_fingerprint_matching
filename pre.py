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
OUTPUT_PATH = "f_processed_output_contact/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Function to preprocess an image (Segmentation, Enhancement, Distortion Correction)
def preprocess_image(image_path, output_folder, is_contactless=False):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Rotate contactless images by -90 degrees to align with contact-based fingerprints
    if is_contactless:
        image = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    # Step 1: Segmentation (Placeholder U-Net model can be integrated later)
    segmented = sobel(image)  # Using Sobel filter for edge detection as a placeholder
    segmented = img_as_ubyte(segmented)
    cv2.imwrite(os.path.join(output_folder, "segmented.png"), segmented)
    
    # Step 2: Enhancement (Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(image)
    cv2.imwrite(os.path.join(output_folder, "enhanced.png"), enhanced)
    
    # Step 3: Distortion Correction (Affine + TPS Placeholder)
    h, w = image.shape
    M = np.float32([[1, 0, 5], [0, 1, 5]])  # Simple translation for testing
    transformed = cv2.warpAffine(enhanced, M, (w, h))
    cv2.imwrite(os.path.join(output_folder, "transformed.png"), transformed)
    
    return transformed

# Process all contactless images avoiding overwrites
for folder in os.listdir(CONTACTLESS_PATH):
    folder_path = os.path.join(CONTACTLESS_PATH, folder)
    if os.path.isdir(folder_path):
        output_folder = os.path.join(OUTPUT_PATH, folder)
        os.makedirs(output_folder, exist_ok=True)
        
        for img_name in os.listdir(folder_path):
            if img_name.endswith(".bmp"):  # Ensure only contactless images are processed
                img_path = os.path.join(folder_path, img_name)
                output_img_folder = os.path.join(output_folder, img_name.split('.')[0])
                os.makedirs(output_img_folder, exist_ok=True)
                preprocess_image(img_path, output_img_folder, is_contactless=True)

# # Process all contact-based images
# for img_name in os.listdir(CONTACT_PATH):
#     if img_name.endswith(".jpg"):  # Ensure only contact-based images are processed
#         img_path = os.path.join(CONTACT_PATH, img_name)
#         output_img_folder = os.path.join(OUTPUT_PATH, "contact_based", img_name.split('.')[0])
#         os.makedirs(output_img_folder, exist_ok=True)
#         preprocess_image(img_path, output_img_folder, is_contactless=False)

print("Preprocessing completed. Output stored in", OUTPUT_PATH)
