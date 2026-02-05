import cv2
import numpy as np

def check_image_quality(image_path):
    img = cv2.imread(image_path)
    if img is None:
        return False, "Image could not be read."

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Overexposure / flash
    if gray.mean() > 200:
        return False, "Image is overexposed (flash or glare detected)."

    # Glare detection
    white_ratio = (gray > 240).sum() / gray.size
    if white_ratio > 0.05:
        return False, "Strong glare detected. Reduce reflections and flash."

    # Low contrast
    if gray.std() < 20:
        return False, "Low contrast image. Improve lighting and focus."

    return True, "Image quality acceptable."
