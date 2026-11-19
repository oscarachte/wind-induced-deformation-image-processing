# Wind-Induced Deformation Measurement Using Image Processing

**EPFL · Measurement Techniques · Team project (4 students) · Python / OpenCV**

## Overview

This project investigated the deformation of a flexible strip exposed to airflow from a hair dryer. The experimental setup combined several measurement approaches to study the interaction between airflow and a deformable structure.

My main contribution was developing an **image-based measurement workflow** to track the strip's tip and quantify its displacement over time. I also helped coordinate the project, worked with teammates on measurement methods, and drafted the technical report apart from the sections describing the other three measurement techniques.

![Experimental setup: airflow directed toward a flexible strip](images/frame.jpg)

## Image-processing workflow

1. **Pixel calibration:** `src/pixel.py` lets the user click two points on an image and reports their distance in pixels. A known physical distance can then be used to calculate the image scale.
2. **Tip tracking:** `src/tracking_tige.py` reads a video frame by frame, detects red regions in HSV color space, and selects the largest qualifying contour.
3. **Tip localization:** the point on the detected contour farthest from a manually selected fixed base is treated as the strip tip.
4. **Displacement:** tip movement is calculated relative to its position in the first successfully detected frame, then converted from pixels to millimetres.
5. **Outputs:** the script can export tracking CSVs, a summary, an annotated video, and a displacement-versus-time plot highlighting the maximum detected displacement.

**Important:** the code reports the *magnitude of tip displacement* relative to the initial detected position. This is not a full-field strain measurement. Tracking quality depends on lighting, the red marker, the camera position, and the selected calibration factor.

## Files

```text
src/
  pixel.py            # Interactive two-point pixel distance measurement
  tracking_tige.py    # Video tracking and tip displacement analysis
images/
  frame.jpg           # Experimental setup photo
src/
  frame.jpg           # Calibration input used by pixel.py
  video_tige.mp4      # Original experimental recording
demo/
  tracking_output.mp4 # Annotated tracking demonstration
requirements.txt      # Python dependencies
```

## Running the code

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

For calibration, the included `src/frame.jpg` is read by `pixel.py`. Run:

```bash
python src/pixel.py
```

For video tracking, the original recording is included as `src/video_tige.mp4`. Review `mm_per_pixel` and other settings near the top of the script, then run:

```bash
python src/tracking_tige.py
```

**Video demonstration:** [Original recording](src/video_tige.mp4) · [Annotated tracking output](demo/tracking_output.mp4).

**Note:** the original experimental video is included, but reproducibility still depends on the interactive selections and project-specific calibration settings. The scripts are preserved as used in the project, including interactive OpenCV windows and a macOS-specific command to open the generated video. The calibration factor `0.109 mm/pixel` is a project setting, not a universal value.

## Experimental context

The broader group project also used other experimental measurement techniques. This repository focuses on my image-processing contribution rather than reproducing the complete group experiment.

## Tools and methods

Python · OpenCV · NumPy · pandas · Matplotlib · HSV segmentation · Contour detection · Video processing · Experimental calibration
