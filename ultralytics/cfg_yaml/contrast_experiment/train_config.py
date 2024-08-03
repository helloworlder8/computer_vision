mdoel_yaml1='ultralytics/cfg_yaml/对比实验/yolov3-tiny.yaml'
mdoel_yaml2='ultralytics/cfg_yaml/对比实验/yolov5.yaml'
mdoel_yaml3='ultralytics/cfg_yaml/对比实验/yolov6.yaml'
mdoel_yaml4='ultralytics/cfg_yaml/对比实验/yolov8-C2f-Faster.yaml'
mdoel_yaml5='ultralytics/cfg_yaml/对比实验/yolov8-ghost.yaml'
mdoel_yaml6='ultralytics/cfg_yaml/对比实验/yolov8-p2.yaml'
task_name='detect'

val_interval=1



resume=False
device='0'
epochs=200
batch=200
name1='yolov3-tiny'
name2='yolov5.yaml'
name3='yolov6.yaml'
name4='yolov8-C2f-Faster.yaml'
name5='yolov8-ghost.yaml'
name6='yolov8-p2.yaml'





#resume
mdoel_resume='last-0.891.pt'
name_resume='ass-focus-poolconv-atta-detetct'


# model = YOLO('ultralytics/cfg_yaml_yaml/论文消融实验模型/yolov5.yaml',task_name='detect')
# # ultralytics/cfg_yaml_yaml/论文消融实验模型/light-focus-poolconv-atta-detect.yaml
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