# from ultralytics import YOLO

# # Load a model
# model = YOLO("yolov8n-seg.pt")  # load an official model
# # model = YOLO("path/to/best.pt")  # load a custom model

# # Predict with the model
# results = model("https://ultralytics.com/images/bus.jpg")  # predict on an image
import torch

original_repr = torch.Tensor.__repr__
# 定义自定义的 __repr__ 方法
def custom_repr(self):
    return f'{self.shape} {original_repr(self)}'
    return f'{self.shape}'
# 替换 torch.Tensor 的 __repr__ 方法
torch.Tensor.__repr__ = custom_repr

from ultralytics import YOLO

import sys
import os

# 获取上一级目录
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)

from scripts.utile.free_cuda import get_free_gpu # type: ignore
device = torch.device(f"cuda:{get_free_gpu(start_index=0)}")

# from ultralytics import YOLO


if __name__=="__main__":

    model1 = YOLO("../checkpoints/yolov8n-seg.pt")  # load an official model
    metrics = model1.train(data='ultralytics/cfg/datasets/coco128-seg.yaml',device="0",project="debug",epochs=1,batch=16)


    model2 = YOLO(model='yolov8n-seg.yaml')
    metrics = model2.train(data='ultralytics/cfg/datasets/coco128-seg.yaml',device="0",project="debug",epochs=1,batch=16)

    model3 = YOLO(model='../checkpoints/yolov8n.pt')
    metrics = model3.train(data='ultralytics/cfg/datasets/coco128.yaml',device="0",project="debug",epochs=1,batch=16)
    
    model4 = YOLO(model='yolov8n.yaml')
    metrics = model4.train(data='ultralytics/cfg/datasets/coco128.yaml',device="0",project="debug",epochs=1,batch=16)  
    # model3 = YOLO("runs/segment/train3/weights/last.pt")  # load an official model
    # metrics = model3.train(resume=True,device="0",epochs=3,batch=16)
