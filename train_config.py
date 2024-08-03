
from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
data_str = str(current_directory / Path('../datasets/BIRDSAI-FORE-BACKUP/BIRDSAI-FORE2.yaml'))


mdoel_yaml1='runs/ablation_experiment/ass-0.879/ASS.yaml'
mdoel_yaml2='runs/detect/yolov8-p2.yaml/weights/last.pt'
mdoel_yaml3='runs/detect/yolov8-R1/weights/last.pt'
mdoel_yaml4='ultralytics/cfg_yaml/ablation_experiment/ASS-newfocus-poolconv-atta.yaml'
mdoel_yaml5='ultralytics/cfg_yaml/ablation_experiment/ASS-newfocus-poolconv-atta-detect.yaml'

name1='yolov8-p2'
name2=''
name3=''
name4='ass-focus-poolconv-atta'
name5='ass-focus-poolconv-atta-detetct'
# 批 模 名
batch1=10
batch2=130
batch3=130
batch4=18
batch5=18



task_name='detect'

val_interval=1
resume=False
device='2'
epochs=200


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