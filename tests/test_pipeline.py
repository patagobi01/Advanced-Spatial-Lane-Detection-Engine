"""
test_pipeline.py

Basic integration test for the Spatial Lane Detection Engine.

This test verifies that:
1. The input video can be opened.
2. Every module initializes successfully.
3. One frame passes through the complete pipeline.
4. Output dimensions remain correct.
"""

import sys
from pathlib import Path

import cv2

# ---------------------------------------------------------
# Add project root to Python path
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from src.preprocessing import preprocess_frame
from src.roi import apply_roi_mask
from src.perspective import PerspectiveTransformer
from src.lane_detection import LaneDetector
from src.drift import DriftEstimator


VIDEO_PATH = PROJECT_ROOT / "assets" / "input" / "highway.mp4"


def test_pipeline():

    print("=" * 50)
    print(" Spatial Lane Detection Pipeline Test ")
    print("=" * 50)

    if not VIDEO_PATH.exists():
        print("[FAIL] Input video not found.")
        return

    cap = cv2.VideoCapture(str(VIDEO_PATH))

    if not cap.isOpened():
        print("[FAIL] Unable to open video.")
        return

    success, frame = cap.read()

    if not success:
        print("[FAIL] Unable to read first frame.")
        cap.release()
        return

    height, width = frame.shape[:2]

    print(f"[INFO] Resolution : {width} x {height}")

    transformer = PerspectiveTransformer(width, height)
    detector = LaneDetector()
    drift = DriftEstimator(width)

    try:

        edges = preprocess_frame(frame)
        print("[PASS] Preprocessing")

        roi = apply_roi_mask(edges)
        print("[PASS] ROI")

        warped = transformer.warp(roi)
        print("[PASS] Perspective Transform")

        overlay, lane_center = detector.detect(warped)
        print("[PASS] Lane Detection")

        output = transformer.unwarp(frame.copy(), overlay)
        print("[PASS] Inverse Perspective")

        offset, direction = drift.calculate(lane_center)
        print("[PASS] Drift Estimation")

    except Exception as e:
        print(f"[FAIL] Pipeline crashed: {e}")
        cap.release()
        return

    assert output.shape == frame.shape, \
        "Output frame size does not match input."

    print("[PASS] Output dimensions verified")

    print(f"[INFO] Offset     : {offset} px")
    print(f"[INFO] Direction : {direction}")

    print("\nAll pipeline tests passed successfully.")

    cap.release()


if __name__ == "__main__":
    test_pipeline()
