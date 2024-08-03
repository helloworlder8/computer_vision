import warnings
warnings.filterwarnings('ignore')
from ultralytics import YOLO

if __name__ == '__main__':
    # choose your yaml file
    # model = YOLO('ultralytics/cfg_yaml/test_model_yaml/ShuffleNet_24_04_04.3_lightcodattention_Detect_Efficient.yaml',task_name='detect')
    # model = YOLO('ultralytics/cfg_yaml/test_model_yaml/ShuffleNet_24_04_04.3_lightcodattention.yaml',task_name='detect')
    model = YOLO('ultralytics/cfg_yaml/ablation_experiment/ASS-focus-poolconv-atta-iou.yaml',task_name='detect')
    # model = YOLO('ultralytics/cfg_yaml/ablation_experiment/yolov8.yaml')
    model.info(detailed=True)
    model.profile(imgsz=[640, 640])
    model.fuse()

# Model summary (fused): 159 layers, 1415078 parameters, 1415062 gradients
# Model summary (fused): 168 layers, 3151904 parameters, 3151888 gradients, 8.7 GFLOPs