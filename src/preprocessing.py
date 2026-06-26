"""
preprocessing.py

This module performs all image preprocessing before
lane detection.

Pipeline:
1. Convert to grayscale
2. Normalize brightness
3. Gaussian Blur
4. Canny Edge Detection
"""

import cv2
import numpy as np


# --------------------------------------------------
# Adjustable Parameters
# --------------------------------------------------

GAUSSIAN_KERNEL = (5, 5)

CANNY_LOW = 50
CANNY_HIGH = 150

CLAHE_CLIP_LIMIT = 2.0
CLAHE_GRID_SIZE = (8, 8)


# --------------------------------------------------
# Internal Functions
# --------------------------------------------------

def _to_grayscale(frame: np.ndarray) -> np.ndarray:
    """
    Convert BGR frame to grayscale.
    """
    return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)


def _normalize_brightness(gray: np.ndarray) -> np.ndarray:
    """
    Improve contrast using CLAHE.

    Helps in:
    - Shadows
    - Evening roads
    - Slight overexposure
    """

    clahe = cv2.createCLAHE(
        clipLimit=CLAHE_CLIP_LIMIT,
        tileGridSize=CLAHE_GRID_SIZE,
    )

    return clahe.apply(gray)


def _gaussian_blur(gray: np.ndarray) -> np.ndarray:
    """
    Remove high-frequency noise.
    """

    return cv2.GaussianBlur(
        gray,
        GAUSSIAN_KERNEL,
        0,
    )


def _detect_edges(image: np.ndarray) -> np.ndarray:
    """
    Detect lane edges using Canny.
    """

    return cv2.Canny(
        image,
        CANNY_LOW,
        CANNY_HIGH,
    )


# --------------------------------------------------
# Public Pipeline
# --------------------------------------------------

def preprocess_frame(frame: np.ndarray) -> np.ndarray:
    """
    Complete preprocessing pipeline.

    Parameters
    ----------
    frame : np.ndarray
        Original BGR image.

    Returns
    -------
    np.ndarray
        Binary edge image.
    """

    gray = _to_grayscale(frame)

    normalized = _normalize_brightness(gray)

    blurred = _gaussian_blur(normalized)

    edges = _detect_edges(blurred)

    return edges
