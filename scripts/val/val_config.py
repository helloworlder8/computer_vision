

from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
# data = str(current_directory / Path('../../../datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml'))
# data = str(current_directory / Path('../../../datasets/ISOD1.1/ISOD.yaml'))
data = "coco128.yaml"

model_yaml1='../checkpoints/yolov8n.pt'
model_yaml2='runs/ISOD-datasets/ALSSs-CA-ISOD-0.798/weights/best.pt'
model_yaml3='runs/ISOD-datasets/ALSSm-CA-ISOD-0.770/weights/best.pt'
model_yaml4='runs/contrast_experiment/yolov8-p2-0.900/weights/best.pt'
model_yaml5='plot/ass-focus-poolconv-atta-iou-0.891_best.pt'
model_yaml6='plot/yolov8-width_best.pt'

project="ISOD-datasets-val"
name1='ALSSn-CA-ISOD-val'
name2='ALSSs-CA-ISOD-val'
name3='ALSSm-CA-ISOD-val'
name4='yolov8-p2-0.900'
name5='ass-focus-poolconv-atta-iou-0.891'
name6='yolov8-width'
# 批 模 名
device='2'