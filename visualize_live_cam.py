import os
import cv2

# Assuming `ultralytics` is a placeholder for the actual way to import and use YOLO.
# Make sure to replace this with the correct import statement for your YOLO model,
# such as `from yolov5 import YOLO` if you're using the Ultralytics implementation.
from ultralytics import YOLO

# No need for VIDEOS_DIR for live feed

# Use 0 for the default camera, or replace with another index or video source if needed
cap = cv2.VideoCapture(0)

# Check if the webcam is opened correctly
if not cap.isOpened():
    raise IOError("Cannot open webcam")

ret, frame = cap.read()
if not ret:
    print("Failed to grab a frame")
    exit()

H, W, _ = frame.shape

# Assuming your YOLO model is correctly loaded
model_path = os.path.join("lp_train", "train", "weights", "best.pt")
model = YOLO(model_path)  # load a custom model
threshold = 0.5

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Inference
    results = model(frame)[0]

    # Draw bounding boxes and labels on each detection
    for result in results.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = result
        if score > threshold:
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

    # Display the frame
    cv2.imshow("Live Object Detection", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Release the VideoCapture object and close display window
cap.release()
cv2.destroyAllWindows()
