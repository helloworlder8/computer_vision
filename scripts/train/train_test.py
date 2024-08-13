import warnings
warnings.filterwarnings('ignore')
from scripts.train.train_config import *
from ultralytics import YOLO
import requests
def send_notice(content):
    token = "853672d072e640479144fba8b29b314b"
    title = "训练成功"
    url = f"http://www.pushplus.plus/send?token={token}&title={title}&content={content}&template=html"
    response = requests.request("GET", url)
    print(response.text)


if __name__ == '__main__':
    model = YOLO(model='yolov8n.yaml',task="detect")

    metrics = model.train(data="ultralytics/cfg/datasets/coco128.yaml",
                cache=False,
                imgsz=640,
                # close_mosaic=10,
                # workers=4,
                # optimizer='SGD', # using SGD
                resume=resume, # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                task="detect",
                project='',
                name="test",
                device= "0",
                epochs=3,
                batch=2,
                
                patience = patience,

                )
    send_notice(f"Precision(B): {metrics.results_dict['metrics/mAP50(B)']}, "
      f"Recall(B): {metrics.results_dict['metrics/mAP50-95(B)']}, "
      f"mAP50(B): {metrics.results_dict['metrics/mAP50(M)']}, "
      f"mAP50-95(B): {metrics.results_dict['metrics/mAP50-95(M)']}, "
      f"Fitness: {metrics.results_dict['fitness']}")
    






