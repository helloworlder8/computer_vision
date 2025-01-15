

from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
data = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-20.yaml'))
# data = "coco128.yaml"
model_yaml1='RM_RDD_yolov8x-world-24-11-21-fine-turn.pt'
model_yaml2='/home/easyits/ang/RM-RDD_COMP_EXP/yolov9s_0.653/weights/best.pt'
model_yaml3='/home/easyits/ang/RM-RDD_COMP_EXP/yolov10/weights/best.pt'
# model_yaml1='/home/easyits/ang/RM-RDD_COMP_EXP/yolov3s.yaml/weights/best.pt'
# model_yaml2='/home/easyits/ang/RM-RDD_COMP_EXP/yolov5s.yaml/weights/best.pt'
# model_yaml3='/home/easyits/ang/RM-RDD_COMP_EXP/yolov6s.yaml/weights/best.pt'
model_yaml4='/home/easyits/ang/RM-RDD_COMP_EXP/yolov8m-ghost.yaml/weights/best.pt'
model_yaml5='/home/easyits/ang/RM-RDD_COMP_EXP/yolov8s-p23/weights/best.pt'
model_yaml6='/home/easyits/ang/RM-RDD_COMP_EXP/yolov9s_0.653/weights/best.pt'
model_yaml7='/home/easyits/ang/RM-RDD_COMP_EXP/yolov10/weights/best.pt'
model_yaml8='/home/easyits/ang/RM-RDD_COMP_EXP/yolov11s.yaml/weights/best.pt'
# model_yaml5='run/BANANA-NEW/ALSSn-seg-MSCAM-last_0.916/weights/last.pt'

project="temp"
name1='RM_RDD_yolov8x-world-24-11-21-fine-turn.pt'
name2='M1'
name3='M2'
name4='M3'
name5='M4'
name6='M5'
name7='M6'
name8='v8'
# 批 模 名
device='0'

#  CIoU or DIoU or EIoU or SIoU or FineSIoU or WIoU
IoU1 = "CIoU"
IoU2 = "CIoU"
IoU3 = "CIoU"
IoU4 = "CIoU"
IoU5 = "CIoU"
IoU6 = "CIoU"
IoU7 = "CIoU"
IoU8 = "CIoU"


# from pathlib import Path

# # 获取当前脚本的相对路径
# current_file_path = Path(__file__)
# current_directory = current_file_path.parent
# # data = str(current_directory / Path('../../../datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml'))
# data = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))
# # data = "coco128.yaml"

# model_yaml1='run/Comparative_experiment_exp/yolov3-tiny-seg_0.782/weights/last.pt'
# model_yaml2='run/Comparative_experiment_exp/yolov5-seg_0.844/weights/last.pt'
# model_yaml3='run/Comparative_experiment_exp/yolov6-seg_0.858/weights/last.pt'
# model_yaml4='run/Comparative_experiment_exp/yolov8-seg-ghost_0.849/weights/last.pt'
# model_yaml5='run/Comparative_experiment_exp/yolov8-seg-p2_0.851/weights/last.pt'
# model_yaml6='run/Comparative_experiment_exp/yolov8-seg-p6_0.851/weights/last.pt'

# model_yaml7='run/Comparative_experiment_exp/yolov9t-seg_0.849/weights/last.pt'
# model_yaml8='run/Comparative_experiment_exp/yolov10-seg_0.848/weights/last.pt'
# # model_yaml5='run/BANANA-NEW/ALSSn-seg-MSCAM-last_0.916/weights/last.pt'

# project="BANANA_val"
# name1='yolov3'
# name2='yolov5'
# name3='yolov6'
# name4='yolov8-seg-ghost'
# name5='yolov8-seg-p2'
# name6='yolov8-seg-p6'
# name7='yolov9t-seg'
# name8='yolov10-seg'
# # 批 模 名
# device='1'




# from pathlib import Path

# # 获取当前脚本的相对路径
# current_file_path = Path(__file__)
# current_directory = current_file_path.parent
# data = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-FINE.yaml'))


# model_yaml1='/home/easyits/ang/RM-RDD_EXP/yolo11x4/weights/best.pt'
# model_yaml2="result/Comparative_experiment_exp/yolov8-seg-p6_0.851/weights/last.pt"
# model_yaml3='result/Comparative_experiment_exp/yolov6-seg_0.858/weights/last.pt'
# model_yaml4='result/Comparative_experiment_exp/yolov8-seg-ghost_0.849/weights/last.pt'
# model_yaml5='result/Ablation_experiment_exp/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt'
# model_yaml6='result/Ablation_experiment_exp/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt'
# model_yaml7='result/Ablation_experiment_exp/ALSSn-seg-24-MSCAMv3-last_0.858/weights/last.pt'


# model_yaml8='run/Comparative_experiment_exp/yolov10-seg_0.848/weights/last.pt'
# # model_yaml5='run/BANANA-NEW/ALSSn-seg-MSCAM-last_0.916/weights/last.pt'


# project="../RM-RDD_EXP"
# name1="yolo11x-val"

# name2="yolov8-seg-p6"
# name3="yolov6-seg"
# name4='yolov8-seg-ghost'
# name5='ALSSn'
# name6='M5'
# name7='M6'
# name8='yolov10-seg'
# # 批 模 名
# device='0'

# #  CIoU or DIoU or EIoU or SIoU or FineSIoU or WIoU
# IoU1 = "CIoU"
# IoU2 = "CIoU"
# IoU3 = "CIoU"
# IoU4 = "CIoU"
# IoU5 = "CIoU"
# IoU6 = "CIoU"
# IoU7 = "CIoU"
# IoU8 = "CIoU"