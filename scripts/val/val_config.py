

from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
data = str(current_directory / Path('../../../datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml'))


model_yaml1='runs/detect/ALSSn-BIRDSAI/weights/best.pt'
model_yaml2='results/ISOD数据集训练实验/ALSSs-ISOD/weights/best.pt'
model_yaml3='results/ISOD数据集训练实验/ALSSm-ISOD/weights/best.pt'
model_yaml4='runs/contrast_experiment/yolov8-p2-0.900/weights/best.pt'
model_yaml5='plot/ass-focus-poolconv-atta-iou-0.891_best.pt'
model_yaml6='plot/yolov8-width_best.pt'


name1='ALSSn-BIRDSAI'
name2='ALSSs-ISOD'
name3='ALSSm-ISOD'
name4='yolov8-p2-0.900'
name5='ass-focus-poolconv-atta-iou-0.891'
name6='yolov8-width'
# 批 模 名
device='2'
