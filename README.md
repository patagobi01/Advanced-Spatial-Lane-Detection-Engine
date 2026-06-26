# Advanced Spatial Lane Detection Engine

A real-time lane detection system built with Python, OpenCV, and NumPy that detects road lane boundaries from dashcam footage and estimates vehicle drift using classical computer vision techniques.

Instead of relying on deep learning, this project uses perspective transformation, edge detection, region-of-interest masking, and the Probabilistic Hough Transform to build a lightweight perception pipeline capable of running in real time on a standard CPU.

The project was developed to explore the computer vision fundamentals behind lane perception systems used in Advanced Driver Assistance Systems (ADAS).

---

## Demo

> Add a GIF here

![Demo](docs/demo.gif)

---

## Results

| Original Frame | Bird's Eye View |
|----------------|-----------------|
| *(Screenshot)* | *(Screenshot)* |

| Edge Detection | Final Output |
|----------------|--------------|
| *(Screenshot)* | *(Screenshot)* |

---

## What this project does

For every video frame, the pipeline performs the following operations:

1. Converts the frame to grayscale
2. Enhances local contrast using CLAHE
3. Reduces image noise with Gaussian Blur
4. Extracts edges using the Canny detector
5. Removes irrelevant regions using a trapezoidal ROI mask
6. Applies a perspective transform to obtain a top-down road view
7. Detects lane markings using the Probabilistic Hough Transform
8. Estimates the left and right lane boundaries
9. Computes the lane center and vehicle offset
10. Projects the detected lane back onto the original frame

---

## Example Output

```
FPS           : 47.3
Lane Offset   : -18 px
Direction     : LEFT
```

---

## Project Structure

```
Spatial-Lane-Detection-Engine
│
├── assets/
├── src/
├── tests/
├── main.py
├── requirements.txt
└── README.md
```

---

## Algorithms Used

| Stage | Algorithm |
|--------|-----------|
| Contrast Enhancement | CLAHE |
| Noise Removal | Gaussian Blur |
| Edge Extraction | Canny Edge Detector |
| ROI Extraction | Polygon Masking |
| Perspective Mapping | Homography |
| Lane Detection | Probabilistic Hough Transform |
| Lane Estimation | Averaged Line Model |
| Vehicle Position | Lane Center Estimation |

---

## Performance

Tested on

- Intel Core i5-1135G7
- Intel Iris Xe Graphics
- 8 GB RAM

| Resolution | Performance |
|------------|-------------|
| 720p | ~40–60 FPS |
| 1080p | ~20–30 FPS |

---

## Running the Project

Clone the repository

```bash
git clone https://github.com/<your-username>/Spatial-Lane-Detection-Engine.git
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run

```bash
python main.py
```

---

## Future Work

The current implementation focuses on classical computer vision. Some natural extensions include:

- Sliding-window lane tracking
- Polynomial lane fitting
- Curved lane detection
- Temporal smoothing
- Camera calibration
- Lane confidence estimation
- Deep-learning-based lane segmentation

---

## Acknowledgements

This project was built for learning and experimentation with real-time computer vision and geometric transformations using OpenCV and NumPy.
