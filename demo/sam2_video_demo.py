# from ultralytics import SAM

# # Load a model
# model = SAM("sam2.1_b.pt")

# # Display model information (optional)
# model.info()

# # Run inference
# model("demo/bedroom.mp4")


from ultralytics.models.sam import SAM2VideoPredictor

# Create SAM2VideoPredictor
overrides = dict(conf=0.25, task="segment", mode="predict", imgsz=1024, model_name="sam2.1_t.pt")
predictor = SAM2VideoPredictor(overrides=overrides)

# Run inference with single point
# results = predictor(source="demo/bedroom.mp4", points=[920, 470], labels=1)

# Run inference with multiple points
results = predictor(source="demo/bedroom.mp4", points=[[920, 470], [909, 138]], labels=[1, 1]) #一张图片两个点,两个标签

# Run inference with multiple points prompt per object
results = predictor(source="demo/bedroom.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 1]])

# Run inference with negative points prompt
results = predictor(source="demo/bedroom.mp4", points=[[[920, 470], [909, 138]]], labels=[[1, 0]])