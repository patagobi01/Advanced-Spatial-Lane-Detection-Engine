"""
drift.py

Calculates the vehicle's lateral offset from the
center of the detected lane and provides a steering
recommendation.

This module assumes the camera is mounted near the
center of the vehicle.
"""

from typing import Tuple


class DriftEstimator:

    def __init__(self, frame_width: int):

        self.frame_center = frame_width // 2

        # Pixel tolerance before considering drift
        self.dead_zone = 25

    # ---------------------------------------------------------

    def calculate(self, lane_center: int) -> Tuple[int, str]:
        """
        Parameters
        ----------
        lane_center : int
            Estimated center of the detected lane.

        Returns
        -------
        offset : int
            Horizontal offset in pixels.

        direction : str
            Steering recommendation.
        """

        offset = lane_center - self.frame_center

        direction = self._get_direction(offset)

        return offset, direction

    # ---------------------------------------------------------

    def _get_direction(self, offset: int) -> str:

        if abs(offset) <= self.dead_zone:
            return "STRAIGHT"

        if offset < 0:
            return "LEFT"

        return "RIGHT"

    # ---------------------------------------------------------

    def steering_angle(self, offset: int) -> float:
        """
        Approximate steering angle.

        Returns
        -------
        float
            Steering angle in degrees.

        Positive  -> Right
        Negative  -> Left
        """

        max_angle = 30.0

        angle = (offset / self.frame_center) * max_angle

        if angle > max_angle:
            angle = max_angle

        elif angle < -max_angle:
            angle = -max_angle

        return round(angle, 2)
