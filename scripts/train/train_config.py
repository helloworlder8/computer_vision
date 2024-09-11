
from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
# data = str(current_directory / Path('../../../datasets/Javeri_det_seg/Javeri_detect_in.yaml'))
data1 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA.yaml'))
data2 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA.yaml'))
data3 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA.yaml'))
data4 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA.yaml'))

data5 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA.yaml'))
data6 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA.yaml'))
data7 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA.yaml'))
data8 = str(current_directory / Path('../../../datasets/BANANA1.1/BANANA.yaml'))
data = "coco-seg.yaml"
# Javeri_det_seg/Javeri_detect+50%.yaml

model_yaml1='ultralytics/cfg_yaml/models/ALSS2/ALSSn-seg-24-MSCAM.yaml'
model_yaml2='ultralytics/cfg_yaml/models/ALSS2/ALSSn-seg-24-MSCAM.yaml'
model_yaml3="ultralytics/cfg_yaml/models/ALSS2/ALSSn-seg-24-MSCAM.yaml"
model_yaml4='ultralytics/cfg_yaml/models/ALSS2/ALSSn-seg-24-MSCAM.yaml'
model_yaml5='ultralytics/cfg_yaml/models/ALSS2/ALSSn-seg-24-MSCAM.yaml'
model_yaml6='ultralytics/cfg_yaml/models/ALSS2/ALSSn-seg-24-MSCAM.yaml'
model_yaml7='ultralytics/cfg_yaml/models/ALSS2/ALSSn-seg-24-MSCAM.yaml'
model_yaml8='ultralytics/cfg_yaml/models/ALSS2/ALSSn-seg-24-MSCAM.yaml'
task='segment'
# task='detect'

project="../exp/BANANA1.1_test"

name1="ALSSn-seg-24-MSCAM-CIoU"
name2='ALSSn-seg-24-MSCAM-DIoU'
name3="ALSSn-seg-24-MSCAM-EIoU"
name4='ALSSn-seg-24-MSCAM-SIoU'

name5='ALSSn-seg-24-MSCAM-FineSIoU'
name6='ALSSn-seg-24-MSCAM-WIoU'
name7='ALSSn-seg-24-MSCAM-FineSIoUv2'
name8='Javeri_detect'

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
IoU2 = "DIoU"
IoU3 = "EIoU"
IoU4 = "SIoU"
IoU5 = "FineSIoU"
IoU6 = "WIoU"
IoU7 = "FineSIoUv2"
IoU8 = "CIoU"
val_interval=1
resume=False
device='0'
epochs=130
patience=100






# from pathlib import Path

# # 获取当前脚本的相对路径
# current_file_path = Path(__file__)
# current_directory = current_file_path.parent
# data = str(current_directory / Path('../../../datasets/Javeri_det_seg/Javeri_segment.yaml'))
# # data = str(current_directory / Path('../../../datasets/ISOD1.1/ISOD.yaml'))
# # data = "coco128.yaml"


# model_yaml1='yolov8n-seg.yaml'
# model_yaml2='ultralytics/cfg/models/ALSS/ALSSs.yaml'
# model_yaml3="ultralytics/cfg/models/ALSS/ALSSm.yaml"
# model_yaml4='ultralytics/cfg/models/ALSS/add_contrast_experiment/ALSSn-CA.yaml'
# model_yaml5='ultralytics/cfg/models/ALSS/add_contrast_experiment/ALSSs-CA.yaml'
# model_yaml6='ultralytics/cfg/models/ALSS/add_contrast_experiment/ALSSm-CA.yaml'

# name1="Javeri-seg"
# name2='ALSSs'
# name3="ALSSm"
# name4='ALSSn-CA'
# name5='ALSSs-CA'
# name6='ALSSm-CA'

# IoU = "CIoU"
# project=""
# # 批 模 名
# batch1=-1
# batch2=-1
# batch3=-1
# batch4=-1
# batch5=-1
# batch6=-1

# val_interval=1

# task='segment'
# # task='detect'

# resume=False
# device='0'
# epochs=200
# patience=20