import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.util import img_as_ubyte
from skimage.filters import gaussian

# Define paths
CONTACT_PATH = "fingerprints/contact-based_fingerprints/first_session"
OUTPUT_PATH = "f_processed_contact_based/"
os.makedirs(OUTPUT_PATH, exist_ok=True)

# Function to enhance fingerprint and extract minutiae points
def extract_minutiae(image_path, output_folder):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None
    
    # Flip image horizontally to correct orientation
    image = cv2.flip(image, 1)
    
    # Apply Gaussian Blur to reduce noise
    image = gaussian(image, sigma=1)
    image = (image * 255).astype(np.uint8)
    
    # Step 1: Binarization
    _, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    
    # Step 2: Skeletonization
    skeleton = skeletonize(binary // 255)
    skeleton = img_as_ubyte(skeleton)
    cv2.imwrite(os.path.join(output_folder, "skeleton.png"), skeleton)
    
    # Step 3: Minutiae Extraction (Advanced Algorithm using Ridge Ending Detection)
    minutiae_points = []
    h, w = skeleton.shape
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            if skeleton[i, j] == 255:
                neighbors = np.sum(skeleton[i-1:i+2, j-1:j+2] > 0)
                if neighbors == 2:  # Likely a ridge ending
                    minutiae_points.append((j, i))
    
    # Save the extracted minutiae points
    np.savetxt(os.path.join(output_folder, "minutiae.txt"), minutiae_points, fmt='%d')
    
    return minutiae_points

# Process all contact-based images
for img_name in os.listdir(CONTACT_PATH):
    if img_name.endswith(".jpg"):  # Ensure only contact-based images are processed
        img_path = os.path.join(CONTACT_PATH, img_name)
        output_img_folder = os.path.join(OUTPUT_PATH, img_name.split('.')[0])
        os.makedirs(output_img_folder, exist_ok=True)
        extract_minutiae(img_path, output_img_folder)

print("Minutiae extraction for contact-based fingerprints completed. Output stored in", OUTPUT_PATH)
