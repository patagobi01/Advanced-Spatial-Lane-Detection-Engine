"""
utils.py

Utility functions used throughout the project.

Contains:
- FPS calculation
- Information overlay
- Text rendering
- Status color selection
"""

import time
import cv2


# ---------------------------------------------------------
# FPS Calculation
# ---------------------------------------------------------

def calculate_fps(previous_time):
    """
    Calculate real-time FPS.

    Parameters
    ----------
    previous_time : float or None

    Returns
    -------
    fps : float
    previous_time : float
    """

    current_time = time.time()

    if previous_time is None:
        return 0.0, current_time

    delta = current_time - previous_time

    if delta <= 0:
        return 0.0, current_time

    fps = 1.0 / delta

    return round(fps, 2), current_time


# ---------------------------------------------------------
# Status Color
# ---------------------------------------------------------

def _direction_color(direction):

    if direction == "STRAIGHT":
        return (0, 255, 0)

    if direction == "LEFT":
        return (0, 165, 255)

    return (0, 0, 255)


# ---------------------------------------------------------
# Draw Text
# ---------------------------------------------------------

def _put_text(frame, text, x, y, color):

    cv2.putText(
        frame,
        text,
        (x, y),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        color,
        2,
        cv2.LINE_AA,
    )


# ---------------------------------------------------------
# Information Overlay
# ---------------------------------------------------------

def draw_information(
    frame,
    fps,
    offset,
    direction,
):
    """
    Draw all information on the output frame.
    """

    color = _direction_color(direction)

    cv2.rectangle(
        frame,
        (10, 10),
        (340, 140),
        (40, 40, 40),
        -1,
    )

    cv2.rectangle(
        frame,
        (10, 10),
        (340, 140),
        color,
        2,
    )

    _put_text(
        frame,
        "Spatial Lane Detection",
        25,
        40,
        (255, 255, 255),
    )

    _put_text(
        frame,
        f"FPS : {fps:.2f}",
        25,
        70,
        (255, 255, 255),
    )

    _put_text(
        frame,
        f"Offset : {offset:+d} px",
        25,
        100,
        (255, 255, 255),
    )

    _put_text(
        frame,
        f"Direction : {direction}",
        25,
        130,
        color,
    )

    height, width = frame.shape[:2]

    center_x = width // 2

    cv2.line(
        frame,
        (center_x, height),
        (center_x, height - 80),
        (255, 255, 255),
        2,
    )

    cv2.circle(
        frame,
        (center_x + offset, height - 40),
        8,
        color,
        -1,
    )

    cv2.line(
        frame,
        (center_x, height - 40),
        (center_x + offset, height - 40),
        color,
        2,
    )


# ---------------------------------------------------------
# Debug Helper
# ---------------------------------------------------------

def stack_frames(left, right):
    """
    Stack two frames horizontally for debugging.
    """

    return cv2.hconcat([left, right])
