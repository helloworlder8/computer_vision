

from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
data = str(current_directory / Path('../../../datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml'))


model_yaml1='runs/ALSS-BIRDSAI-CA/ALSSm-CA-BIRDSAI-0.890/weights/best.pt'
model_yaml2='runs/ALSS-BIRDSAI-CA/ALSSn-CA-BIRDSAI-0.880/weights/best.pt'
model_yaml3='runs/ALSS-BIRDSAI-CA/ALSSs-CA-BIRDSAI-0.889/weights/best.pt'
model_yaml4='runs/contrast_experiment/yolov8-p2-0.900/weights/best.pt'
model_yaml5='plot/ass-focus-poolconv-atta-iou-0.891_best.pt'
model_yaml6='plot/yolov8-width_best.pt'


name1='ALSSm-CA-BIRDSAI'
name2='ALSSn-CA-BIRDSAI'
name3='ALSSs-CA-BIRDSAI'
name4='yolov8-p2-0.900'
name5='ass-focus-poolconv-atta-iou-0.891'
name6='yolov8-width'
# 批 模 名
device='2'