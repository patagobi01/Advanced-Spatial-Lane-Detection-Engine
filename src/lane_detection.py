"""
lane_detection.py

Detects road lanes using Probabilistic Hough Transform.

Pipeline
--------
1. Detect line segments
2. Reject horizontal/vertical noise
3. Separate left/right lanes
4. Average lane lines
5. Draw lane overlay
6. Estimate lane center
"""

import cv2
import numpy as np


class LaneDetector:

    def __init__(self):

        self.hough_threshold = 40
        self.min_line_length = 40
        self.max_line_gap = 150

        self.min_slope = 0.45

    # -------------------------------------------------

    def detect(self, bird_view):

        height, width = bird_view.shape

        lines = cv2.HoughLinesP(
            bird_view,
            rho=1,
            theta=np.pi / 180,
            threshold=self.hough_threshold,
            minLineLength=self.min_line_length,
            maxLineGap=self.max_line_gap
        )

        overlay = np.zeros((height, width, 3), dtype=np.uint8)

        if lines is None:
            return overlay, width // 2

        left_lines = []
        right_lines = []

        for line in lines:

            x1, y1, x2, y2 = line[0]

            if x2 == x1:
                continue

            slope = (y2 - y1) / (x2 - x1)

            if abs(slope) < self.min_slope:
                continue

            if slope < 0:
                left_lines.append((x1, y1, x2, y2))

            else:
                right_lines.append((x1, y1, x2, y2))

        left_lane = self._average_lane(left_lines, height)
        right_lane = self._average_lane(right_lines, height)

        lane_center = width // 2

        if left_lane is not None:

            cv2.line(
                overlay,
                left_lane[0],
                left_lane[1],
                (0, 255, 0),
                8,
            )

        if right_lane is not None:

            cv2.line(
                overlay,
                right_lane[0],
                right_lane[1],
                (0, 255, 0),
                8,
            )

        if left_lane is not None and right_lane is not None:

            polygon = np.array([
                left_lane[0],
                left_lane[1],
                right_lane[1],
                right_lane[0]
            ], dtype=np.int32)

            cv2.fillPoly(
                overlay,
                [polygon],
                (0, 180, 0)
            )

            bottom_left = left_lane[0][0]
            bottom_right = right_lane[0][0]

            lane_center = (bottom_left + bottom_right) // 2

        return overlay, lane_center

    # -------------------------------------------------

    def _average_lane(self, lines, image_height):

        if len(lines) == 0:
            return None

        slopes = []
        intercepts = []

        for x1, y1, x2, y2 in lines:

            slope = (y2 - y1) / (x2 - x1)

            intercept = y1 - slope * x1

            slopes.append(slope)
            intercepts.append(intercept)

        slope = np.mean(slopes)
        intercept = np.mean(intercepts)

        y_bottom = image_height
        y_top = int(image_height * 0.60)

        x_bottom = int((y_bottom - intercept) / slope)
        x_top = int((y_top - intercept) / slope)

        return (
            (x_bottom, y_bottom),
            (x_top, y_top)
        )
