from ultralytics import YOLO
import os


# Load model
model_path = os.path.join("lp_train", "train", "weights", "last.pt")
model = YOLO(model_path)  # load a custom model

results = model(
    "/home/smebellis/CIS583_ALPR_Final_project_3-24-2024/data/test/images/00a7d31c6cc6b7f3_jpg.rf.2707e63f5c51f113de704441ea210a65.jpg"
)

# Process results and display

for result in results:

    result.save(filename="output.jpg")
