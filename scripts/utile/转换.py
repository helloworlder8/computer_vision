from ultralytics import YOLO

# Load a model
model = YOLO("yolov8s-seg.pt")  # load an official model
# model = YOLO("path/to/best.pt")  # load a custom trained model

# Export the model
model.export(format="ncnn")