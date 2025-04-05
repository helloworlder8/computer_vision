

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

data1 = "VOC.yaml"
data2 = "VOC.yaml"


model_yaml1="ultralytics/cfg_yaml/models/ALSS-YOLO-world/ALSS-YOLO-world.yaml"
model_yaml2="ultralytics/cfg_yaml/models/v8/yolov8-worldv2.yaml"

model_yaml3="yolov6s.yaml"
model_yaml4='yolov8s.yaml'
model_yaml5='yolov9s.yaml'
model_yaml6='yolov11s.yaml'
model_yaml7='yolov8s-p2.yaml'
model_yaml8='yolov8m-ghost.yaml'

# task='segment'
task='detect'

project="../EXP"

name1="ALSS-YOLO-world-VOC"
name2="yolov8-worldv2-VOC"

name3="yolov6s.yaml"
name4='yolov8s.yaml'
name5='yolov9s.yaml'
name6='yolov11s.yaml'
name7='yolov8s-p2.yaml'
name8='yolov8m-ghost.yaml'


# 批 模 名
batch1=-1
batch2=-1
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


val_interval=1
resume=False
device='0'
epochs=100
patience=30



