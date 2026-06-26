"""
roi.py

Applies a Region of Interest (ROI) mask to remove
unnecessary parts of the frame.

Only the road region is preserved for further
processing.

The ROI is generated dynamically based on the
frame dimensions.
"""

import cv2
import numpy as np


def _create_roi_vertices(width: int, height: int) -> np.ndarray:
    """
    Create trapezoidal ROI vertices.

    The polygon adapts automatically to
    different video resolutions.
    """

    vertices = np.array([
        [
            (int(width * 0.10), height),
            (int(width * 0.45), int(height * 0.60)),
            (int(width * 0.55), int(height * 0.60)),
            (int(width * 0.90), height),
        ]
    ], dtype=np.int32)

    return vertices


def apply_roi_mask(edge_image: np.ndarray) -> np.ndarray:
    """
    Apply ROI masking.

    Parameters
    ----------
    edge_image : np.ndarray
        Binary edge image.

    Returns
    -------
    np.ndarray
        Masked edge image.
    """

    height, width = edge_image.shape

    mask = np.zeros_like(edge_image)

    vertices = _create_roi_vertices(width, height)

    cv2.fillPoly(mask, vertices, 255)

    masked = cv2.bitwise_and(edge_image, mask)

    return masked


def draw_roi(frame: np.ndarray) -> np.ndarray:
    """
    Draw the ROI polygon for debugging.

    This function is optional and can be
    used to visualize the selected region.
    """

    output = frame.copy()

    height, width = frame.shape[:2]

    vertices = _create_roi_vertices(width, height)

    cv2.polylines(
        output,
        vertices,
        isClosed=True,
        color=(0, 255, 0),
        thickness=2,
    )

    return output
