import warnings
warnings.filterwarnings('ignore')
from train_config import *
from ultralytics import YOLO
import requests
def send_notice(content):
    token = "853672d072e640479144fba8b29b314b"
    title = "训练成功"
    url = f"http://www.pushplus.plus/send?token={token}&title={title}&content={content}&template=html"
    response = requests.request("GET", url)
    print(response.text)


if __name__ == '__main__':
    model = YOLO(model_str=mdoel_yaml1,task_name=task_name)

    metrics = model.train(data_str=data_str,
                cache=False,
                imgsz=640,
                # close_mosaic=10,
                # workers=4,
                # optimizer='SGD', # using SGD
                val_interval=val_interval,
                resume=resume, # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                task_name=task_name,
                project='',
                device= device,
                epochs=epochs,
                batch=batch1,
                name=name1,
                )
    send_notice(f"Precision(B): {metrics.results_dict['metrics/precision(B)']}, "
      f"Recall(B): {metrics.results_dict['metrics/recall(B)']}, "
      f"mAP50(B): {metrics.results_dict['metrics/mAP50(B)']}, "
      f"mAP50-95(B): {metrics.results_dict['metrics/mAP50-95(B)']}, "
      f"Fitness: {metrics.results_dict['fitness']}")
    






