import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from val_config import *
if __name__ == '__main__':
    model1 = YOLO(mdoel_yaml1)
    model1.val(data_str=data_str,
              split='val',
              imgsz=640,
              batch=16,
              # rect=False,
              # save_json=True, # if you need to cal coco metrice
              device='3',
              project='runs/val',
              name=name1,
              )

    # model2 = YOLO(mdoel_yaml2)
    # model2.val(data_str=data_str,
    #           split='val',
    #           imgsz=640,
    #           batch=16,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device='3',
    #           project='runs/val',
    #           name=name2,
    #           )
    
    # model3 = YOLO(mdoel_yaml3)
    # model3.val(data_str=data_str,
    #           split='val',
    #           imgsz=640,
    #           batch=16,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device='3',
    #           project='runs/val',
    #           name=name3,
    #           )
    
    # model4 = YOLO(mdoel_yaml4)
    # model4.val(data_str=data_str,
    #           split='val',
    #           imgsz=640,
    #           batch=16,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device='3',
    #           project='runs/val',
    #           name=name4,
    #           )
    
    # model5 = YOLO(mdoel_yaml5)
    # model5.val(data_str=data_str,
    #           split='val',
    #           imgsz=640,
    #           batch=16,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device='3',
    #           project='runs/val',
    #           name=name5,
    #           )
    
    # model6 = YOLO(mdoel_yaml6)
    # model6.val(data_str=data_str,
    #           split='val',
    #           imgsz=640,
    #           batch=16,
    #           # rect=False,
    #           # save_json=True, # if you need to cal coco metrice
    #           device='3',
    #           project='runs/val',
    #           name=name6,
    #           )