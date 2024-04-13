import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt
import easyocr
from util import read_license_plate
import argparse


reader = easyocr.Reader(["en"], gpu=True)

# Argument parser that takes in an image path
parser = argparse.ArgumentParser(description="Image Path Argument Parser")

# Add an argument for the image path
parser.add_argument("image_path", type=str, help="Path to the image file")

# Parse the command line arguments
args = parser.parse_args()

# Get the image path from the parsed arguments
image_path = args.image_path

frame = cv2.imread(image_path)
frame_copy = frame.copy()

# CHeck if image was opened correctly
if frame is None:
    raise IOError("Cannont open image")

H, W, _ = frame.shape

# Load the model weights
model_path = os.path.join("lp_train", "train", "weights", "best.pt")
model = YOLO(model_path)  # load a custom model
threshold = 0.5

# Inference
results = model(frame)[0]

# Draw bounding boxes and labels on each detection
for result in results.boxes.data.tolist():
    x1, y1, x2, y2, score, class_id = result
    if score > threshold:
        width = x2 - x1
        height = y2 - y1

        # Calculate a certain percentage of the width and height
        new_width = width * 0.90
        new_height = height * 0.5

        # Calculate the new coordinates
        x1 = x1 + (width - new_width) / 2
        y1 = y1 + (height - new_height) / 2
        x2 = x1 + new_width
        y2 = y1 + new_height

        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(
            frame,
            results.names[int(class_id)].upper(),
            (int(x1), int(y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (0, 255, 0),
            2,
        )

        license_plate_crop = frame[int(y1) : int(y2), int(x1) : int(x2)]

        height, width, _ = license_plate_crop.shape
        center_x, center_y = width // 2, height // 2
        # Read the license plate text
        detections = reader.readtext(license_plate_crop)
        for detection in detections:
            bbox, text, score = detection

            text = text.upper().replace(" ", "")
            if text:
                cv2.putText(
                    frame_copy,
                    text,
                    ((int(center_x), int(center_y - 1))),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.9,
                    (0, 255, 0),
                    2,
                )

    plt.imshow(frame_copy)
    plt.show()
