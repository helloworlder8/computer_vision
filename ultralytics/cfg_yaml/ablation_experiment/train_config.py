mdoel_yaml1='ultralytics/cfg_yaml/ablation_experiment/ASS-newfocus-poolconv-atta-iou.yaml'
mdoel_yaml2='ultralytics/cfg_yaml/ablation_experiment/ASS-newfocus-poolconv-atta-heavy.yaml'
mdoel_yaml3='ultralytics/cfg_yaml/ablation_experiment/yolov5-width.yaml'
mdoel_yaml4='ultralytics/cfg_yaml/ablation_experiment/ASS-newfocus-poolconv-atta.yaml'
mdoel_yaml5='ultralytics/cfg_yaml/ablation_experiment/ASS-newfocus-poolconv-atta-detect.yaml'
task_name='detect'

val_interval=1



resume=False
device='2'
epochs=230
batch=180
name1='ASS-newfocus-poolconv-atta-iou'
name2='ASS-newfocus-poolconv-atta-heavy'
name3='yolov5-width.yaml'
name4='ass-focus-poolconv-atta'
name5='ass-focus-poolconv-atta-detetct'




#resume
mdoel_yaml6='runs/detect/ass-focus-poolconv-atta5/weights/last.pt'
name6='ass-focus-poolconv-atta-detetct'


# model = YOLO('ultralytics/cfg_yaml/论文消融实验模型/yolov5.yaml',task_name='detect')
# # ultralytics/cfg_yaml/论文消融实验模型/light-focus-poolconv-atta-detect.yaml
# metrics = model.train(data_str="../../datasets/BIRDSAI-FORE-BACKUP/BIRDSAI-FORE.yaml",
#             cache=False,
#             imgsz=640,
#             # close_mosaic=10,
#             # workers=4,
#             # optimizer='SGD', # using SGD
#             val_interval=1,
#             resume='true', # last.pt path
#             # amp=False # close amp
#             # fraction=0.2,
#             task_name='detect',
#             project='',
#             # device='2',
#             epochs=200,
#             batch=2,
#             name='