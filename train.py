from ultralytics import RTDETR

# Load a COCO-pretrained RT-DETR-l model
model = RTDETR("rtdetr-l.pt")

# Display model information (optional)
model.info()

# Train the model on the COCO8 example dataset for 100 epochs
results = model.train(data="coco8.yaml", epochs=10, imgsz=640)

# Run inference with the RT-DETR-l model on the 'bus.jpg' image
results = model("ultralytics/assets/bus.jpg")