import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from scripts.val.val_config import *
if __name__ == '__main__':
    model1 = YOLO(model_yaml1)
    model1.val(data=data,
              split='val',
              imgsz=640,
              batch=20,
              IoU=IoU1,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              device=device,
              project=project,
              name=name1,
              )

    # model2 = YOLO(model_yaml2)
    # model2.val(data=data,
    #           split='val',
    #           imgsz=640,
    #           batch=1,
    #           IoU=IoU2,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device=device,
    #           project=project,
    #           name=name2,
    #           )
    
    # model3 = YOLO(model_yaml3)
    # model3.val(data=data,
    #           split='val',
    #           imgsz=640,
    #           batch=1,
    #           IoU=IoU3,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device=device,
    #           project=project,
    #           name=name3,
    #           )
    
    # model4 = YOLO(model_yaml4)
    # model4.val(data=data,
    #           split='val',
    #           imgsz=640,
    #           batch=1,
    #           IoU=IoU4,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device=device,
    #           project=project,
    #           name=name4,
    #           )
    
    # model5 = YOLO(model_yaml5)
    # model5.val(data=data,
    #           split='val',
    #           imgsz=640,
    #           batch=1,
    #           IoU=IoU5,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device=device,
    #           project=project,
    #           name=name5,
    #           )
    
    # model6 = YOLO(model_yaml6)
    # model6.val(data=data,
    #           split='val',
    #           imgsz=640,
    #           batch=1,
    #           IoU=IoU6,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device=device,
    #           project=project,
    #           name=name6,
    #           )

    # model7 = YOLO(model_yaml7)
    # model7.val(data=data,
    #           split='val',
    #           imgsz=640,
    #           batch=1,
    #           IoU=IoU7,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device=device,
    #           project=project,
    #           name=name7,
    #           )
    
    # model8 = YOLO(model_yaml8)
    # model8.val(data=data,
    #           split='val',
    #           imgsz=640,
    #           batch=1,
    #           IoU=IoU8,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device=device,
    #           project=project,
    #           name=name8,
    #           )
    # # 2.3