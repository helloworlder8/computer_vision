  
    
import warnings
warnings.filterwarnings('ignore')
from scripts.predict.preditct_config import *
from ultralytics import YOLO
from scripts.utile.send_notice import send_notice_by_task


if __name__ == '__main__':
    model1 = YOLO(model=model_yaml1)
    model1.predict(source=source,
                  fraction=fraction,
                  imgsz=640,
                  project=project,
                  name=name1,
                  save=True,
                  device=device,
                  batch=batch,
                  ch=3,
                  conf = 0.3,
                  show_labels = show_labels,
                  save_txt = True
                #   visualize=True # visualize model features maps
                )
# 0000000352_0000000000_0000000155.jpg
    # model2 = YOLO(model=model_yaml2)
    # model2.predict(source=source,
    #               imgsz=640,
    #               project=project,
    #               name=name2,
    #               save=True,
    #               device=device,
    #               batch=batch,
    #               ch=3,
    #               conf = 0.3,
    #               show_labels = show_labels
    #             #   visualize=True # visualize model features maps
    #             )
    # model3 = YOLO(model=model_yaml3)
    # model3.predict(source=source,
    #               imgsz=640,
    #               project=project,
    #               name=name3,
    #               save=True,
    #               device=device,
    #               batch=batch,
    #               ch=3,
    #               conf = 0.3,
    #               show_labels = show_labels
    #             #   visualize=True # visualize model features maps
    #             )
    
    # model4 = YOLO(model=model_yaml4)
    # model4.predict(source=source,
    #               imgsz=640,
    #               project=project,
    #               name=name4,
    #               save=True,
    #               device=device,
    #               batch=batch,
    #               ch=3,
    #               conf = 0.3,
    #               show_labels = show_labels
    #             #   visualize=True # visualize model features maps
    #             )
    # model5 = YOLO(model=model_yaml5)
    # model5.predict(source=source,
    #               imgsz=640,
    #               project=project,
    #               name=name5,
    #               save=True,
    #               device=device,
    #               batch=batch,
    #               ch=3,
    #               conf = 0.3,
    #               show_labels = show_labels
    #             #   visualize=True # visualize model features maps
    #             )
    
    # model6 = YOLO(model=model_yaml6)
    # model6.predict(source=source,
    #               imgsz=640,
    #               project=project,
    #               name=name6,
    #               save=True,
    #               device=device,
    #               batch=batch,
    #               ch=3,
    #               conf = 0.3,
    #               show_labels = show_labels
    #             #   visualize=True # visualize model features maps
    #             )
    
    # model7 = YOLO(model=model_yaml7)
    # model7.predict(source=source,
    #               imgsz=640,
    #               project=project,
    #               name=name4,
    #               save=True,
    #               device=device,
    #               batch=batch,
    #               ch=3,
    #               conf = 0.3,
    #               show_labels = show_labels
    #             #   visualize=True # visualize model features maps
    #             )
    # model8 = YOLO(model=model_yaml8)
    # model8.predict(source=source,
    #               imgsz=640,
    #               project=project,
    #               name=name4,
    #               save=True,
    #               device=device,
    #               batch=batch,
    #               ch=3,
    #               conf = 0.3,
    #               show_labels = show_labels
    #             #   visualize=True # visualize model features maps
    #             )
    # send_notice_by_task(None, None)