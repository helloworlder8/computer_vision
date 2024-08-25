
from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
# data = str(current_directory / Path('../../../datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml'))
# data = str(current_directory / Path('../../../datasets/ISOD1.1/ISOD.yaml'))
data = "coco128.yaml"


model_yaml1='../checkpoints/yolov8n.pt'
model_yaml2='ultralytics/cfg/models/ALSS/ALSSs.yaml'
model_yaml3="ultralytics/cfg/models/ALSS/ALSSm.yaml"
model_yaml4='ultralytics/cfg/models/ALSS/add_contrast_experiment/ALSSn-CA.yaml'
model_yaml5='ultralytics/cfg/models/ALSS/add_contrast_experiment/ALSSs-CA.yaml'
model_yaml6='ultralytics/cfg/models/ALSS/add_contrast_experiment/ALSSm-CA.yaml'

name1="ALSSn"
name2='ALSSs'
name3="ALSSm"
name4='ALSSn-CA'
name5='ALSSs-CA'
name6='ALSSm-CA'

IoU = "FineSIoU"
project="ISOD-datasets"
# 批 模 名
batch1=2
batch2=-1
batch3=-1
batch4=-1
batch5=-1
batch6=-1

val_interval=1

task='detect'

resume=False
device='0'
epochs=1
patience=10