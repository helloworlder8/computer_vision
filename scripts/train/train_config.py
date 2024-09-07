
from pathlib import Path

# 获取当前脚本的相对路径
current_file_path = Path(__file__)
current_directory = current_file_path.parent
# data = str(current_directory / Path('../../../datasets/Javeri_det_seg/Javeri_detect_in.yaml'))
data1 = str(current_directory / Path('../../../datasets/ADE20K_2016_yolo/ADE20K_detect+10%.yaml'))
data2 = str(current_directory / Path('../../../datasets/ADE20K_2016_yolo/ADE20K_detect+20%.yaml'))
data3 = str(current_directory / Path('../../../datasets/ADE20K_2016_yolo/ADE20K_detect+50%.yaml'))
data4 = str(current_directory / Path('../../../datasets/ADE20K_2016_yolo/ADE20K_detect.yaml'))

data5 = str(current_directory / Path('../../../datasets/Javeri_det_seg/Javeri_detect+10%.yaml'))
data6 = str(current_directory / Path('../../../datasets/Javeri_det_seg/Javeri_detect+20%.yaml'))
data7 = str(current_directory / Path('../../../datasets/Javeri_det_seg/Javeri_detect+50%.yaml'))
data8 = str(current_directory / Path('../../../datasets/Javeri_det_seg/Javeri_detect.yaml'))
# data = "coco-seg.yaml"
# Javeri_det_seg/Javeri_detect+50%.yaml

model_yaml1='yolov8x.yaml'
model_yaml2='yolov8x.yaml'
model_yaml3="yolov8x.yaml"
model_yaml4='yolov8x.yaml'
model_yaml5='yolov8x.yaml'
model_yaml6='yolov8x.yaml'
model_yaml7='yolov8x.yaml'
model_yaml8='yolov8x.yaml'
# task='segment'
task='detect'

project="ADE20K_detect"
name1="ADE20K_detect+10%"
name2='ADE20K_detect+20%'
name3="ADE20K_detect+50%"
name4='ADE20K_detect'

name5='Javeri_detect+10%'
name6='Javeri_detect+20%'
name7='Javeri_detect+50%'
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

IoU = "CIoU"
val_interval=1
resume=False
device='1'
epochs=200
patience=30






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