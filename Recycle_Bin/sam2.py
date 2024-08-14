from ultralytics import SAM

# Load a model
model = SAM("sam2_b.pt")

# Display model information (optional)
model.info()

# Segment with bounding box prompt
results = model("ultralytics/assets/bus.jpg", bboxes=[100, 100, 200, 200])

# Segment with point prompt
# results = model("path/to/image.jpg", points=[150, 150], labels=[1])