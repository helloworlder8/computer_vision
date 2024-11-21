""" YOLOWorld """
# from ultralytics import YOLOWorld
# # Initialize a YOLO-World model
# model = YOLOWorld("../checkpoints/yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes
# # Execute inference with the YOLOv8s-world model on the specified image
# results = model.predict("ultralytics/assets/bus_640.jpg")
# # Show results
# results[0].show()



# from ultralytics.models.yolo.detect import DetectionPredictor
# args = dict(model_name='../checkpoints/yolov8s-worldv2.pt')
# predictor = DetectionPredictor(overrides=args)
# predictor("ultralytics/assets/bus_640.jpg")



from ultralytics import YOLO
# Initialize a YOLO-World model
model = YOLO("../checkpoints/yolov8x-worldv2.pt")  # or select yolov8m/l-world.pt
# Define custom classes
model.set_classes(["Potholes", "Sink", 'Vertical cracks', 'Patch Vertical cracks', 'Horizontal cracks', 'Patching horizontal cracks', 'Network crack', 'Patch Network crack','cockroach'])

# Save the model with the defined offline vocabulary
model.save("RM_RDD_cockroach.pt")