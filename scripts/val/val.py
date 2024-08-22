import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from scripts.val.val_config import *
if __name__ == '__main__':
    model1 = YOLO(model_yaml1)
    model1.val(data=data,
              split='val',
              imgsz=640,
              batch=2,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              device=device,
              project='',
              name=name1,
              )

    model2 = YOLO(model_yaml2)
    model2.val(data=data,
              split='val',
              imgsz=640,
              batch=64,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              device=device,
              project='',
              name=name2,
              )
    
    model3 = YOLO(model_yaml3)
    model3.val(data=data,
              split='val',
              imgsz=640,
              batch=64,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              device=device,
              project='',
              name=name3,
              )