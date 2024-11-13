  
    
import warnings
warnings.filterwarnings('ignore')
from scripts.predict.preditct_config import *
from ultralytics import YOLO
from scripts.utile.send_notice import send_notice_by_task


if __name__ == '__main__':
    model1 = YOLO(model=model_yaml1)

    model1.predict(source=source,
                  imgsz=640,
                  project='runs/predict',
                  name=name1,
                  save=True,
                  device=device,
                  batch=batch,
                  ch=1,
                  conf = 0.6,
                  show_labels = show_labels
                #   visualize=True # visualize model features maps
                )

    # send_notice_by_task(metrics,task)