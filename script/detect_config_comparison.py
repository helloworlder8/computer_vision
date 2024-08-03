
from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
data_str = str(current_directory / Path('../../datasets/BIRDSAI-FORE-BACKUP/BIRDSAI-FORE.yaml'))


model_yaml1='plot/comparative_experiment/yolov5-0.876_best.pt'
model_yaml2='plot/comparative_experiment/yolov8-ghost-0.879_best.pt'
model_yaml3='plot/comparative_experiment/yolov8_AM_0.889_best.pt'
model_yaml4='plot/comparative_experiment/yolov8-p2-0.900_best.pt'
model_yaml5='plot/ablation_experiment/ass-focus-poolconv-atta-iou-0.891_best.pt'
model_yaml6='plot/comparative_experiment/ASS-YOLO-best.pt'


name1='yolov5'
name2='yolov8-ghost'
name3='yolov8_AM'
name4='yolov8-p2'
name5='ASS-YOLO'
name6='ASS-YOLO-m'
# 批 模 名
device='2'

source = 'select_img/3'

batch = 16

show_labels = True

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