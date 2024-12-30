

from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
data1 = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-FINE.yaml'))
data2 = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-FINE.yaml'))
data3 = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-FINE.yaml'))
data4 = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-FINE.yaml'))
data5 = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-FINE.yaml'))
data6 = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-FINE.yaml'))
data7 = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-FINE.yaml'))
data8 = str(current_directory / Path('../../../datasets/RM-RDD/RM-RDD-FINE.yaml'))


# data1 = "coco8.yaml"
# data2 = "coco8.yaml"
# data3 = "coco8.yaml"
# data4 = "coco8.yaml"
# data5 = "coco8.yaml"
# data6 = "coco8.yaml"
# data7 = "coco8.yaml"
# data8 = "coco8.yaml"

#yolov3s.yaml 10,256,616
#yolov5s.yaml 9,114,632
#yolov6s.yaml 10,920,488
#yolov8s.yaml 11,128,680
#yolov9s.yaml 7,170,184
#yolov11s.yaml 10,713,816
#yolov8s-p2.yaml 10,093,448
#yolov8m-ghost.yaml 10,279,74
#yolov8m-ghost-p2.yaml 8,988,064

model_yaml1="yolov10s.yaml"
model_yaml2="yolov8s-p2.yaml"

model_yaml3="yolov6s.yaml"
model_yaml4='yolov8s.yaml'
model_yaml5='yolov9s.yaml'
model_yaml6='yolov11s.yaml'
model_yaml7='yolov8s-p2.yaml'
model_yaml8='yolov8m-ghost.yaml'

# task='segment'
task='detect'

project="../RM-RDD_COMP_EXP"

name1="yolov10"
name2="yolov8s-p2"

name3="yolov6s.yaml"
name4='yolov8s.yaml'
name5='yolov9s.yaml'
name6='yolov11s.yaml'
name7='yolov8s-p2.yaml'
name8='yolov8m-ghost.yaml'


# 批 模 名
batch1=-1
batch2=16
batch3=-1
batch4=-1
batch5=-1
batch6=-1

batch7=-1
batch8=-1


#  CIoU or DIoU or EIoU or SIoU or FineSIoU or WIoU
IoU1 = "CIoU"
IoU2 = "CIoU"
IoU3 = "CIoU"
IoU4 = "CIoU"
IoU5 = "CIoU"
IoU6 = "CIoU"
IoU7 = "CIoU"
IoU8 = "CIoU"


val_interval=5
resume=False
device='0'
epochs=300
patience=30









# from pathlib import Path

# # 获取当前脚本的相对路径
# current_file_path = Path(__file__)
# current_directory = current_file_path.parent
# # data = str(current_directory / Path('../../../datasets/Javeri_det_seg/Javeri_detect_in.yaml'))
# data1 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))
# data2 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))
# data3 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))
# data4 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))

# data5 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))
# data6 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))
# data7 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))
# data8 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA1.1.yaml'))
# data = "coco-seg.yaml"

# model_yaml1="run/Ablation_experiment/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt"
# model_yaml2="run/Ablation_experiment/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt"
# model_yaml3="run/Ablation_experiment/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt"
# model_yaml4='run/Ablation_experiment/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt'
# model_yaml5='run/Ablation_experiment/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt'
# model_yaml6='run/Ablation_experiment/ALSSn-seg-24-MSCAMv3_0.857/weights/last.pt'
# model_yaml7='run/Comparative_experiment/yolov9t-seg.yaml'
# model_yaml8='run/Comparative_experiment/yolov10-seg.yaml'

# task='segment'
# # task='detect'
# # GIoU=False, DIoU=False, CIoU=False, EIoU=False, SIoU=False
# project="run/Ablation_experiment_exp"

# name1="GIoU"
# name2="DIoU"
# name3="CIoU"
# name4='EIoU'
# name5='SIoU'


# name6='WIoU'

# name7='yolov9t-seg'
# name8='yolov10-seg'

# # 批 模 名
# batch1=-1
# batch2=-1
# batch3=-1
# batch4=-1
# batch5=-1
# batch6=-1
# batch7=-1
# batch8=-1
# #  CIoU or DIoU or EIoU or SIoU or FineSIoU or WIoU
# IoU1 = "GIoU"
# IoU2 = "DIoU"
# IoU3 = "EIoU"
# IoU4 = "CIoU"
# IoU5 = "SIoU"
# IoU6 = "WIoU"
# IoU7 = "CIoU"



# IoU8 = "CIoU"
# val_interval=10
# resume=True
# device='1'
# epochs=230
# patience=200
