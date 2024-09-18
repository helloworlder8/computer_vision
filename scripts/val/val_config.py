

from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
# data = str(current_directory / Path('../../../datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml'))
data = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))
# data = "coco128.yaml"

model_yaml1='run/result/yolov8-width-seg2_0.867/weights/last.pt'
model_yaml2='run/result/ALSSn-withoutcov-seg_0.865/weights/last.pt'
model_yaml3='run/result/ALSSn-seg_0.866/weights/last.pt'
model_yaml4='run/result/ALSSn-seg-24_0.867/weights/last.pt'
model_yaml5='run/result/ALSSn-seg-24-MSCAMv2_0.856/weights/last.pt'
model_yaml6='run/result/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt'

model_yaml7='run/result/ALSSn-seg-24-MSCAMv3-last_0.858/weights/last.pt'
model_yaml8='run/9-18-1/ALSSn-seg-24-MSCAMv8_0.858/weights/last.pt'
# model_yaml5='run/BANANA-NEW/ALSSn-seg-MSCAM-last_0.916/weights/last.pt'

project="BANANA_val"
name1='yolov8-seg-1'
name2='M1'
name3='M2'
name4='M3'
name5='M4'
name6='M5'
name7='M6'
name8='v8'
# 批 模 名
device='1'