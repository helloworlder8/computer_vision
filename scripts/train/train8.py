import warnings
warnings.filterwarnings('ignore')
from scripts.train.train_config import *
from ultralytics import YOLO
from scripts.utile.send_notice import send_notice_by_task


if __name__ == '__main__':
    model = YOLO(model=model_yaml8,task=task)

    metrics = model.train(data=data8,
                batch=batch8,
                IoU=IoU8,
                cache=False,
                imgsz=640,
                
                # close_mosaic=10,
                # workers=4,
                # optimizer='SGD', # using SGD

                val_interval=val_interval,
                resume=resume, # last.pt path
                # amp=False # close amp
                # fraction=0.2,
                task=task,
                project=project,
                name=name8,
                device= device,
                epochs=epochs,

                
                patience = patience,

                )

    send_notice_by_task(metrics,task)






# Model summary: 251 layers, 1,577,473 parameters, 1,577,457 gradients, 8.2 GFLOPs