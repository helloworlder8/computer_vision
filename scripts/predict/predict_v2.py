   
  
    
import warnings
warnings.filterwarnings('ignore')
from scripts.predict.preditct_config import *
from ultralytics import YOLO
from scripts.utile.send_notice import send_notice_by_task


if __name__ == '__main__':
    model1 = YOLO("yolov8s-seg.pt")

    model1.predict(source="scripts/assets/zidane_640.jpg",
                  imgsz=640,
                  project='runs/predict',
                  name=name1,
                  save=True,
                  device=device,
                  batch=batch,
                  ch=3,
                  conf = 0.3,
                  show_labels = show_labels
                #   visualize=True # visualize model features maps
                )

    # send_notice_by_task(metrics,task)