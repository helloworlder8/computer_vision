
from pathlib import Path

# # 获取当前脚本的相对路径
# current_file_path = Path(__file__)
# current_directory = current_file_path.parent
# data = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-20.yaml'))
# # datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml

model_yaml1='Gradio-YOLO/checkpoints/ALSS-YOLO-seg.pt'
model_yaml2='/home/easyits/ang/exp/RM-RDD_COMP_EXP/yolov6s.yaml/weights/best.pt'
model_yaml3='/home/easyits/ang/exp/RM-RDD_COMP_EXP/yolov8s.yaml/weights/best.pt'
model_yaml4='/home/easyits/ang/exp/RM-RDD_COMP_EXP/yolov9s_0.653/weights/best.pt'
model_yaml5='/home/easyits/ang/exp/RM-RDD_COMP_EXP/yolov10/weights/best.pt'
model_yaml6='/home/easyits/ang/exp/RM-RDD_COMP_EXP/yolov11s.yaml/weights/best.pt'

model_yaml7='/home/easyits/ang/RM-RDD_COMP_EXP/yolov10/weights/best.pt'
model_yaml8='/home/easyits/ang/RM-RDD_COMP_EXP/yolov11s.yaml/weights/best.pt'
fraction = 1

project='temp'
name1='yolov5s'
name2='yolov6s'
name3='yolov8S'
name4='yolov9s'
name5='yolov10s'
name6='yolov11s'

name7='yolov10s'
name8='yolov11s'
# 批 模 名
device='0'

source = 'Gradio-YOLO/img_examples/0000000353_0000000000_0000000652.jpg'

batch = 20

show_labels = True