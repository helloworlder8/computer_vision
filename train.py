# from ultralytics import RTDETR

# # Load a COCO-pretrained RT-DETR-l model
# model = RTDETR("../checkpoints/rtdetr-l.pt")

# # Display model information (optional)
# model.info()

# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=1, imgsz=640)

# # Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("ultralytics/assets/bus.jpg")


import warnings
warnings.filterwarnings('ignore')

from ultralytics import YOLO


if __name__ == '__main__':
    model = YOLO(model="yolov8.yaml",task="detect")

    metrics = model.train(data="coco128.yaml",
                batch=2,
                cache=False,
                imgsz=640,
                
                # close_mosaic=10,
                # workers=4,
                # optimizer='SGD', # using SGD


                resume=False, # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                task="detect",
                project="",
                name="",
                device= "0",
                epochs=10,

                
                patience = 30,

                )
