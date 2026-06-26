import cv2
from pathlib import Path

from src.preprocessing import preprocess_frame
from src.roi import apply_roi_mask
from src.perspective import PerspectiveTransformer
from src.lane_detection import LaneDetector
from src.drift import DriftEstimator
from src.utils import (
    calculate_fps,
    draw_information,
)

# ============================================================
# CONFIGURATION
# ============================================================

INPUT_VIDEO = "assets/input/highway.mp4"
OUTPUT_VIDEO = "assets/output/output.mp4"

WINDOW_NAME = "Spatial Lane Detection"

# ============================================================


def main():

    input_path = Path(INPUT_VIDEO)
    output_path = Path(OUTPUT_VIDEO)

    if not input_path.exists():
        print(f"[ERROR] Video not found: {input_path}")
        return

    output_path.parent.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(input_path))

    if not cap.isOpened():
        print("[ERROR] Unable to open video.")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fps = cap.get(cv2.CAP_PROP_FPS)

    if fps <= 1:
        fps = 30

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )

    transformer = PerspectiveTransformer(width, height)
    detector = LaneDetector()
    drift = DriftEstimator(width)

    previous_time = None

    print("\n======================================")
    print(" Spatial Lane Detection Engine Started ")
    print("======================================")
    print(f"Resolution : {width} x {height}")
    print(f"FPS        : {fps:.0f}")
    print("Press 'Q' to quit.\n")

    while True:

        success, frame = cap.read()

        if not success:
            break

        # -----------------------------
        # 1. Pre-processing
        # -----------------------------

        edges = preprocess_frame(frame)

        # -----------------------------
        # 2. ROI
        # -----------------------------

        roi = apply_roi_mask(edges)

        # -----------------------------
        # 3. Bird Eye View
        # -----------------------------

        warped = transformer.warp(roi)

        # -----------------------------
        # 4. Lane Detection
        # -----------------------------

        lane_overlay, lane_center = detector.detect(warped)

        # -----------------------------
        # 5. Convert Back
        # -----------------------------

        output = transformer.unwarp(frame, lane_overlay)

        # -----------------------------
        # 6. Drift Calculation
        # -----------------------------

        offset, direction = drift.calculate(lane_center)

        # -----------------------------
        # 7. FPS
        # -----------------------------

        fps_value, previous_time = calculate_fps(previous_time)

        # -----------------------------
        # 8. Information Overlay
        # -----------------------------

        draw_information(
            frame=output,
            fps=fps_value,
            offset=offset,
            direction=direction,
        )

        writer.write(output)

        cv2.imshow(WINDOW_NAME, output)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    writer.release()

    cv2.destroyAllWindows()

    print("\nProcessing Complete.")
    print(f"Saved Output -> {output_path}")


if __name__ == "__main__":
    main()
