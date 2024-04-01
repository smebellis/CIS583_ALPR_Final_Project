from ultralytics import YOLO

# Initialize model
model = YOLO("yolov8n.pt")


# train model

model.train(data="config.yaml", epochs=20, project="lp_train")
