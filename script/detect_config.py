
from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
data_str = str(current_directory / Path('../../datasets/BIRDSAI-FORE-BACKUP/BIRDSAI-FORE.yaml'))


mdoel_yaml1='plot/ass-0.879_best.pt'
mdoel_yaml2='plot/ass-focus-0.887_best.pt'
mdoel_yaml3='plot/ass-focus-poolconv-0.886_best.pt'
mdoel_yaml4='plot/ass-focus-poolconv-atta-iou-0.885_best.pt'
mdoel_yaml5='plot/ass-focus-poolconv-atta-iou-0.891_best.pt'
mdoel_yaml6='plot/yolov8-width_best.pt'


name1='ass-0.879'
name2='ass-focus-0.887'
name3='ass-focus-poolconv-0.886'
name4='ass-focus-poolconv-atta-iou-0.885'
name5='ass-focus-poolconv-atta-iou-0.891'
name6='yolov8-width'
# 批 模 名
device='2'




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