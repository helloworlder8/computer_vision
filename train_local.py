
"""RTDETR 训练"""
# from ultralytics import RTDETR

# # Load a COCO-pretrained RT-DETR-l model
# model = RTDETR("../checkpoints/rtdetr-l.pt")

# # Display model information (optional)
# model.info()

# # Train the model on the COCO8 example dataset for 100 epochs
# results = model.train(data="coco8.yaml", epochs=1,batch=2, imgsz=640)

# # Run inference with the RT-DETR-l model on the 'bus.jpg' image
# results = model("ultralytics/assets/bus.jpg")





"""RTDETR 训练"""
# from ultralytics.models.rtdetr.train import RTDETRTrainer

# args = dict(model_name='rtdetr-l.yaml', data='coco8.yaml', batch=2, imgsz=640, epochs=3)
# trainer = RTDETRTrainer(overrides=args)
# trainer.train()
  
  
"""YOLO 训练"""        
# import warnings
# warnings.filterwarnings('ignore')

# from ultralytics import YOLO


# if __name__ == '__main__':
#     model = YOLO(model="yolov8.yaml",task="detect")

#     metrics = model.train(data="coco128.yaml",
#                 batch=2,
#                 cache=False,
#                 imgsz=640,
                
#                 # close_mosaic=10,
#                 # workers=4,
#                 # optimizer='SGD', # using SGD


#                 resume=False, # last.pt path
#                 # amp=False # close amp
#                 # fraction=0.2,
#                 task="detect",
#                 project="",
#                 name="",
#                 device= "0",
#                 epochs=10,

                
#                 patience = 30,

#                 )




"""YOLO 训练"""
# from ultralytics.models.yolo.detect import DetectionTrainer

# args = dict(model_name='../checkpoints/yolo11n.pt', data='coco8.yaml', epochs=3)
# trainer = DetectionTrainer(overrides=args)
# trainer.train()





# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
# model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights


from ultralytics.models.yolo.classify import ClassificationTrainer

args = dict(model_name='../checkpoints/yolov8n-cls.pt', data='imagenet10', epochs=3)
trainer = ClassificationTrainer(overrides=args)
trainer.train()
        
# Train the model
# results = model.train(data="mnist160", epochs=100, imgsz=64)