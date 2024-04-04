# CIS583_ALPR_Final_Project

A license plate detector trained on YOLOv8 (You Only Look Once, version 8) a deep learning model designed for the specific task of detecting license plates in images or video streams. 

## Prerequisites

Before you begin, ensure you have met the following requirements:
- Python 3.8 or higher
- easyocr
- pandas
- ast
- numpy
- scipy interpolate

## Configuration

Consider using a configuration file for paths and other settings. Edit the provided `config.yml` (or similar) to match your local setup.

## Getting Started

Follow these steps to get your project running:

### Step 1: Update Paths to Model Weights

Ensure that paths to the model weights and video files are set correctly for your environment:

- **Model Paths:**
  - COCO model: `coco_model = YOLO("<path_to_your_model>/yolov8n.pt")`
  - License plate detector: `license_plate_detector = YOLO("<path_to_your_model>/license_plate_detector.pt")`

- **Video Path:** 
  - `cap = cv2.VideoCapture("<path_to_your_video>/sample.mp4")`

### Step 2: Run `main.py`

Execute `main.py` to process the sample video file. This will generate `test.csv`, necessary for the next step.

### Step 3: Generate Interpolated Data

Run `add_missing_data.py` to compute average bounding box locations. This script outputs `test_interpolated.csv`.

### Step 4: Visualize the Results

Ensure paths in `visualize.py` are updated before running:
- CSV Path: `results = pd.read_csv("<path_to_your_data>/test_interpolated.csv")`
- Video Path: `video_path = "<path_to_your_video>/sample.mp4"`

Run `visualize.py` to produce `out.mp4`, showcasing detected license plates.

## Additional Information

The sort module needs to be downloaded from this repository: https://github.com/abewley/sort