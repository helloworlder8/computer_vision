
from pathlib import Path

# # 获取当前脚本的相对路径
# current_file_path = Path(__file__)
# current_directory = current_file_path.parent
# data = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-20.yaml'))
# # datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml

model_yaml1='RM_RDD_yolov8x-world.pt'
model_yaml2='/home/easyits/ang/RM-RDD_COMP_EXP/yolov5s.yaml/weights/best.pt'
model_yaml3='/home/easyits/ang/RM-RDD_COMP_EXP/yolov6s.yaml/weights/best.pt'
model_yaml4='/home/easyits/ang/RM-RDD_COMP_EXP/yolov8m-ghost.yaml/weights/best.pt'
model_yaml5='/home/easyits/ang/RM-RDD_COMP_EXP/yolov8s-p23/weights/best.pt'
model_yaml6='/home/easyits/ang/RM-RDD_COMP_EXP/yolov9s_0.653/weights/best.pt'
model_yaml7='/home/easyits/ang/RM-RDD_COMP_EXP/yolov10/weights/best.pt'
model_yaml8='/home/easyits/ang/RM-RDD_COMP_EXP/yolov11s.yaml/weights/best.pt'
fraction = 1

name1='RM_RDD_yolov8x-world'
name2='yolov5s'
name3='yolov6s'
name4='yolov8m-ghost'
name5='yolov8s-p2'
name6='yolov9s'
name7='yolov10s'
name8='yolov11s'
# 批 模 名
device='0'

source = '/home/easyits/ang/datasets/RM-RDD/train/fine_images'

batch = 20

show_labels = True