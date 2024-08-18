# from ultralytics import FastSAM

# # Define an inference source
# source = "ultralytics/assets/bus.jpg"

# # Create a FastSAM model
# model = FastSAM("FastSAM-s.pt")  # or FastSAM-x.pt

# # Run inference on an image
# # everything_results = model(source, device="cpu", retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)

# # Run inference with bboxes prompt
# results = model(source, bboxes=[439, 437, 524, 709])

# # Run inference with points prompt
# results = model(source, points=[[200, 200]], labels=[1])

# # Run inference with texts prompt
# results = model(source, texts="a photo of a dog")

# # Run inference with bboxes and points and texts prompt at the same time
# results = model(source, bboxes=[439, 437, 524, 709], points=[[200, 200]], labels=[1], texts="a photo of a dog")



from ultralytics.models.fastsam import FastSAMPredictor

# Create FastSAMPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", model_name="FastSAM-s.pt", save=False, imgsz=1024)
predictor = FastSAMPredictor(overrides=overrides)

# Segment everything
everything_results = predictor("scripts/assets")

# Prompt inference
bbox_results = predictor.prompt(everything_results, bboxes=[[200, 200, 300, 300]])
point_results = predictor.prompt(everything_results, points=[200, 200])
# text_results = predictor.prompt(everything_results, texts="a photo of a dog")