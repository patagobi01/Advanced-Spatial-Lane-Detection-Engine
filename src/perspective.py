"""
perspective.py

Performs perspective transformation to obtain a
bird's-eye (top-down) view of the road.

The transformation matrices are calculated once
during initialization and reused for every frame.
"""

import cv2
import numpy as np


class PerspectiveTransformer:

    def __init__(self, width: int, height: int):

        self.width = width
        self.height = height

        self.source = np.float32([
            [
                width * 0.43,
                height * 0.63
            ],
            [
                width * 0.57,
                height * 0.63
            ],
            [
                width * 0.92,
                height * 0.98
            ],
            [
                width * 0.08,
                height * 0.98
            ]
        ])

        self.destination = np.float32([
            [
                width * 0.25,
                0
            ],
            [
                width * 0.75,
                0
            ],
            [
                width * 0.75,
                height
            ],
            [
                width * 0.25,
                height
            ]
        ])

        self.matrix = cv2.getPerspectiveTransform(
            self.source,
            self.destination
        )

        self.inverse_matrix = cv2.getPerspectiveTransform(
            self.destination,
            self.source
        )

    def warp(self, image: np.ndarray) -> np.ndarray:
        """
        Convert image to bird's-eye view.
        """

        warped = cv2.warpPerspective(
            image,
            self.matrix,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR
        )

        return warped

    def unwarp(
        self,
        original_frame: np.ndarray,
        lane_overlay: np.ndarray,
        alpha: float = 0.65
    ) -> np.ndarray:
        """
        Projects the detected lane back onto
        the original frame.
        """

        overlay = cv2.warpPerspective(
            lane_overlay,
            self.inverse_matrix,
            (self.width, self.height),
            flags=cv2.INTER_LINEAR
        )

        result = cv2.addWeighted(
            original_frame,
            1.0,
            overlay,
            alpha,
            0
        )

        return result

    def get_source_points(self) -> np.ndarray:
        """
        Returns source vertices.
        Useful for debugging.
        """
        return self.source.copy()

    def get_destination_points(self) -> np.ndarray:
        """
        Returns destination vertices.
        """
        return self.destination.copy()

    def draw_source_polygon(
        self,
        frame: np.ndarray,
        color=(0, 255, 255),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Draw the perspective source polygon.
        """

        output = frame.copy()

        pts = self.source.astype(np.int32)

        cv2.polylines(
            output,
            [pts],
            isClosed=True,
            color=color,
            thickness=thickness
        )

        return output
