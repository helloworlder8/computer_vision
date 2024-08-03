import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO
from detect_config_comparison import *


if __name__ == '__main__':
    model1 = YOLO(model_yaml1) # select your model.pt path
    model1.predict(source=source,
                  imgsz=640,
                  project='runs/detect',
                  name=name1,
                  save=True,
                  device=device,
                  batch=batch,
                  ch=1,
                  conf = 0.6,
                  show_labels = show_labels
                #   visualize=True # visualize model features maps
                )
            # self.plotted_img = result.plot(
            #     line_width=self.args.line_width,
            #     boxes=self.args.show_boxes,
            #     conf=self.args.show_conf,
            #     labels=self.args.show_labels,
            #     im_gpu=None if self.args.retina_masks else im[i],
            # )  
    model2 = YOLO(model_yaml2) # select your model.pt path
    model2.predict(source=source,
                  imgsz=640,
                  project='runs/detect',
                  name=name2,
                  save=True,
                  device=device,
                  batch=batch,
                  ch=1,
                  conf = 0.6,
                  show_labels = show_labels
                #   visualize=True # visualize model features maps
                )
    
    model3 = YOLO(model_yaml3) # select your model.pt path
    model3.predict(source=source,
                  imgsz=640,
                  project='runs/detect',
                  name=name3,
                  save=True,
                  device=device,
                  batch=batch,
                  ch=1,
                  conf = 0.6,
                  show_labels = show_labels
                #   visualize=True # visualize model features maps
                )
    
    model4 = YOLO(model_yaml4) # select your model.pt path
    model4.predict(source=source,
                  imgsz=640,
                  project='runs/detect',
                  name=name4,
                  save=True,
                  device=device,
                  batch=batch,
                  ch=1,
                  conf = 0.6,
                  show_labels = show_labels
                #   visualize=True # visualize model features maps
                )
    
    model5 = YOLO(model_yaml5) # select your model.pt path
    model5.predict(source=source,
                  imgsz=640,
                  project='runs/detect',
                  name=name5,
                  save=True,
                  device=device,
                  batch=batch,
                  ch=1,
                  conf = 0.6,
                  show_labels = show_labels
                #   visualize=True # visualize model features maps
                )
    
    model6 = YOLO(model_yaml6) # select your model.pt path
    model6.predict(source=source,
                  imgsz=640,
                  project='runs/detect',
                  name=name6,
                  save=True,
                  device=device,
                  batch=batch,
                  ch=1,
                  conf = 0.6,
                  show_labels = show_labels
                #   visualize=True # visualize model features maps
                )