
import warnings
warnings.filterwarnings('ignore')
project="../exp/24-11-21-yolo-world"
name1="RM_RDD_yolov8x-world-24-11-21-head-only-fine-tune"
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
from ultralytics.models.rtdetr.train import RTDETRTrainer

args = dict(model_name='rtdetr-l.yaml', data='coco8.yaml', batch=2, imgsz=640, epochs=3)
trainer = RTDETRTrainer(overrides=args)
trainer.train()
  
  
"""YOLO 训练"""        


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




"""YOLO 训练"""
from ultralytics.models.yolo.detect import DetectionTrainer

args = dict(model_name='../checkpoints/yolov8n.pt', data='coco8.yaml', epochs=1)
trainer = DetectionTrainer(overrides=args)
trainer.train()



# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
# model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-cls.yaml").load("yolo11n-cls.pt")  # build from YAML and transfer weights



"""YOLO 分类"""
# from ultralytics.models.yolo.classify import ClassificationTrainer

# args = dict(model_name='../checkpoints/yolov8n-cls.pt', data='imagenet10', epochs=3)
# trainer = ClassificationTrainer(overrides=args)
# trainer.train()



# from ultralytics import YOLOWorld

# # Initialize a YOLO-World model
# model = YOLOWorld("../checkpoints/yolov8s-world.pt")  # or select yolov8m/l-world.pt for different sizes

# # Execute inference with the YOLOv8s-world model on the specified image
# results = model.predict("path/to/image.jpg")

# # Show results
# results[0].show()



"""YOLOWorld"""
# from ultralytics import YOLOWorld
# model = YOLOWorld("../checkpoints/yolov8s-worldv2.pt")
# results = model.train(data="coco8.yaml", epochs=100, imgsz=640)
# results = model("ultralytics/assets/bus_640.jpg")


"""YOLOWorld"""
# from ultralytics.models.yolo.world import WorldTrainer
# args = dict(model_name='RM_RDD_yolov8x.pt', data='/home/easyits/ang/datasets/RM-RDD/RM-RDD-20.yaml', epochs=300)
# trainer = WorldTrainer(overrides=args) #模型 数据 轮次
# trainer.train()


# from ultralytics.models.yolo.world import WorldTrainer
# args = dict(model_name='/home/easyits/ang/checkpoints/24-11-20-RM_RDD_yolov8x.pt', data='/home/easyits/ang/datasets/RM-RDD/RM-RDD-20.yaml', epochs=300,resume_pt= 'runs/detect/train2/weights/last.pt',resume= False,freeze =22,project=project,name=name1)
# trainer = WorldTrainer(overrides=args) #模型 数据 轮次
# trainer.train()





