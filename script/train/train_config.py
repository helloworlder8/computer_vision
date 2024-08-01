
from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
data = str(current_directory / Path('../../../datasets/banana/banana.yaml'))


model_yaml1='ultralytics/cfg/models/ALSS/ALSSn.yaml'
model_yaml2='ultralytics/cfg/models/ALSS/ALSSs.yaml'
model_yaml3='ultralytics/cfg/models/ALSS/ALSSm.yaml'
model_yaml4='ultralytics/cfg/models/v8/yolov8-seg.yaml'
model_yaml5='ultralytics/cfg/models/ALSS/yolov10n.yaml'

name1='ALSSn-seg'
name2='ALSSs-seg'
name3='ALSSm-seg'
name4='yolov8-seg.yaml'
name5='yolov10n.yaml'
# 批 模 名
batch1=-1
batch2=-1
batch3=-1
batch4=-1
batch5=-1



task='segment'

resume=False
device='0'
epochs=1
patience=200