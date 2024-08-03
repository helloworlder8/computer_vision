import os, tqdm
from ultralytics import YOLO

if __name__ == '__main__':
    error_result = []
    for yaml_path in tqdm.tqdm(os.listdir('ultralytics/cfg_yaml/ablation_experiment')):
        if 'rtdetr' not in yaml_path and 'cls' not in yaml_path:
            try:
                model = YOLO(f'{yaml_path}',task_name='detect')
                # model.info(detailed=True)
                model.profile([640, 640])
                model.fuse()
            except Exception as e:
                error_result.append(f'{yaml_path} {e}')
    
    for i in error_result:
        print(i)