

# from pathlib import Path

# # 获取当前脚本的相对路径
# current_file_path = Path(__file__)
# current_directory = current_file_path.parent
# # data = str(current_directory / Path('../../../datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml'))
# data = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))
# # data = "coco128.yaml"

# model_yaml1='run/result/yolov8-width-seg2_0.867/weights/last.pt'
# model_yaml2='run/result/ALSSn-withoutcov-seg_0.865/weights/last.pt'
# model_yaml3='run/result/ALSSn-seg_0.866/weights/best.pt'
# model_yaml4='run/result/ALSSn-seg-24_0.867/weights/last.pt'
# model_yaml5='run/result/ALSSn-seg-24-MSCAMv2_0.856/weights/last.pt'
# model_yaml6='run/result/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt'

# model_yaml7='run/result/ALSSn-seg-24-MSCAMv3-last_0.858/weights/last.pt'
# model_yaml8='run/9-18-1/ALSSn-seg-24-MSCAMv8_0.858/weights/last.pt'
# # model_yaml5='run/BANANA-NEW/ALSSn-seg-MSCAM-last_0.916/weights/last.pt'

# project="BANANA_val"
# name1='yolov8-seg-1'
# name2='M1'
# name3='M2'
# name4='M3'
# name5='M4'
# name6='M5'
# name7='M6'
# name8='v8'
# # 批 模 名
# device='1'




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




from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
# data = str(current_directory / Path('../../../datasets/BIRDSAI-FORE-BACKUP1.1/BIRDSAI-FORE.yaml'))
data = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))
# data = "coco128.yaml"

model_yaml1='result/Comparative_experiment_exp/yolov9t-seg_0.849/weights/last.pt'
model_yaml2="result/Comparative_experiment_exp/yolov8-seg-p6_0.851/weights/last.pt"
model_yaml3='result/Comparative_experiment_exp/yolov6-seg_0.858/weights/last.pt'
model_yaml4='result/Comparative_experiment_exp/yolov8-seg-ghost_0.849/weights/last.pt'
model_yaml5='result/Ablation_experiment_exp/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt'
model_yaml6='result/Ablation_experiment_exp/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt'
model_yaml7='result/Ablation_experiment_exp/ALSSn-seg-24-MSCAMv3-last_0.858/weights/last.pt'


model_yaml8='run/Comparative_experiment_exp/yolov10-seg_0.848/weights/last.pt'
# model_yaml5='run/BANANA-NEW/ALSSn-seg-MSCAM-last_0.916/weights/last.pt'


project="result"
name1="yolov8s"
name2="yolov8-seg-p6"
name3="yolov6-seg"
name4='yolov8-seg-ghost'
name5='ALSSn'
name6='M5'
name7='M6'

name8='yolov10-seg'
# 批 模 名
device='1'

#  CIoU or DIoU or EIoU or SIoU or FineSIoU or WIoU
IoU1 = "CIoU"
IoU2 = "CIoU"
IoU3 = "CIoU"
IoU4 = "CIoU"
IoU5 = "CIoU"
IoU6 = "CIoU"
IoU7 = "CIoU"
IoU8 = "CIoU"