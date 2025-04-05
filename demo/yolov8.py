

""" predeict """
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("../checkpoints/yolov8n.pt")
# model = YOLO("yolov8.yaml")
# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model("ultralytics/assets/bus.jpg")
results[0].show()







""" train """
from ultralytics import YOLO

# Load a COCO-pretrained YOLOv8n model
model = YOLO("../checkpoints/yolov8n.pt")

#or
# model = YOLO("yolov8.yaml")
# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=100, imgsz=640)

# Run inference with the YOLOv8n model on the 'bus.jpg' image
results = model("ultralytics/assets/bus.jpg")
results[0].show()