
from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
data = str(current_directory / Path('../../../datasets/BIRDSAI-FORE-BACKUP/BIRDSAI-FORE.yaml'))


model_yaml1='runs/detect/ALSSn-CA-BIRDSAIDdatasetSIoU/weights/best.pt'
model_yaml2='plot/ass-focus-0.887_best.pt'
model_yaml3='plot/ass-focus-poolconv-0.886_best.pt'
model_yaml4='plot/ass-focus-poolconv-atta-iou-0.885_best.pt'
model_yaml5='plot/ass-focus-poolconv-atta-iou-0.891_best.pt'
model_yaml6='plot/yolov8-width_best.pt'


name1='ALSSn-CA-BIRDSAIDdatasetSIoU'
name2='ass-focus-0.887'
name3='ass-focus-poolconv-0.886'
name4='ass-focus-poolconv-atta-iou-0.885'
name5='ass-focus-poolconv-atta-iou-0.891'
name6='yolov8-width'
# 批 模 名
device='2'

source = '0000000353_0000000000_0000000891.jpg'

batch = 16

show_labels = True