import warnings
warnings.filterwarnings('ignore')
from scripts.train.train_config import *
from ultralytics import YOLO
from scripts.utile.send_notice import send_notice_by_task



if __name__ == '__main__':
    model = YOLO(model=model_yaml2,task=task)
    # ultralytics/cfg_yaml/论文消融实验模型/light-focus-poolconv-atta-detect.yaml
    metrics = model.train(data=data,
                cache=False,
                imgsz=640,
                # close_mosaic=10,
                # workers=4,
                # optimizer='SGD', # using SGD
                IoU="FineSIoU",
                val_interval=val_interval,
                resume=resume, # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                task=task,
                project='',
                name=name2,
                device=device,
                epochs=epochs,
                batch=batch2,

                patience = patience,
                
                )

    send_notice_by_task(metrics,task)






