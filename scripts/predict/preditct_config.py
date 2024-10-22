
from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
data = str(current_directory / Path('../../../datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml'))
# datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml

model_yaml1='Recycle_Bin/ALSSn-BIRDSAI/weights/best.pt'
model_yaml2='plot/ass-focus-0.887_best.pt'
model_yaml3='plot/ass-focus-poolconv-0.886_best.pt'
model_yaml4='plot/ass-focus-poolconv-atta-iou-0.885_best.pt'
model_yaml5='plot/ass-focus-poolconv-atta-iou-0.891_best.pt'
model_yaml6='plot/yolov8-width_best.pt'


name1='ALSSn-BIRDSAI'
name2='ass-focus-0.887'
name3='ass-focus-poolconv-0.886'
name4='ass-focus-poolconv-atta-iou-0.885'
name5='ass-focus-poolconv-atta-iou-0.891'
name6='yolov8-width'
# 批 模 名
device='0'

source = 'Recycle_Bin'

batch = 2

show_labels = False