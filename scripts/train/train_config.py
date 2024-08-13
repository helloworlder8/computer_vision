
from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
data = str(current_directory / Path('../../../datasets/ISOD/ISOD.yaml'))


model_yaml1='ultralytics/cfg/models/ALSS/ALSSn.yaml'
model_yaml2='ultralytics/cfg/models/ALSS/ALSSs.yaml'
model_yaml3='ultralytics/cfg/models/ALSS/ALSSm.yaml'
model_yaml4='ultralytics/cfg/models/ALSS/add_contrast_experiment/yolov8-ghost.yaml'
model_yaml5='ultralytics/cfg/models/ALSS/add_contrast_experiment/yolov8-p2.yaml'
model_yaml6='ultralytics/cfg/models/ALSS/add_contrast_experiment/ALSSm.yaml'

name1='ALSSn-ISOD'
name2='ALSSs-ISOD'
name3='ALSSm-ISOD'
name4='yolov8-ghost-ISODdataset'
name5='yolov8-p2-ISODdataset'
name6='ALSSm-ISODdataset'

# 批 模 名
batch1=180
batch2=180
batch3=180
batch4=180
batch5=180
batch6=180

val_interval=1

task='detect'

resume=False
device='6'
epochs=200
patience=10