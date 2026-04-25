from __future__ import annotations

from utils import require_cv2

def detect_blur(image, threshold: float = 100.0) -> bool:
    cv2 = require_cv2()
    # Compute the Laplacian of the image and then return the focus measure,
    # which is simply the variance of the Laplacian
    variance = cv2.Laplacian(image, cv2.CV_64F).var()
    return variance < threshold

def detect_darkness(image, threshold: float = 80.0) -> bool:
    import numpy as np
    # Mean pixel intensity
    mean_intensity = np.mean(image)
    return mean_intensity < threshold

def preprocess_plate(crop):
    """
    Adaptive preprocessing for ANPR.
    Handles blur and low-light conditions selectively.
    """
    cv2 = require_cv2()

    # Step 1: Convert to grayscale
    grayscale = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    is_blurry = detect_blur(grayscale)
    is_dark = detect_darkness(grayscale)

    processed = grayscale.copy()

    if is_blurry:
        # Denoise
        processed = cv2.fastNlMeansDenoising(processed, None, h=10, searchWindowSize=21, templateWindowSize=7)
        # Upscale slightly for sharpening
        processed = cv2.resize(processed, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
        # Sharpen using unsharp masking
        gaussian = cv2.GaussianBlur(processed, (9, 9), 10.0)
        processed = cv2.addWeighted(processed, 1.5, gaussian, -0.5, 0)
    elif is_dark:
        # Contrast Limited Adaptive Histogram Equalization
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        processed = clahe.apply(processed)
        # Upscale
        processed = cv2.resize(processed, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    else:
        # Just standard upscale
        processed = cv2.resize(processed, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)

    return processed
