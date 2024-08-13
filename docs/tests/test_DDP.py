# # Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
# overrides = {'task': 'segment', 'mode': 'train', 'model_name': 'yolov8l-seg.yaml', 'data': 'ultralytics/cfg/datasets/coco128-seg.yaml', 'epochs': 3, 'time': None, 'patience': 100, 'batch': 2, 'imgsz': 640, 'save': True, 'save_period': -1, 'cache': False, 'device': '3', 'workers': 8, 'project': None, 'name': 'train3', 'exist_ok': False, 'pretrained': True, 'optimizer': 'auto', 'verbose': True, 'seed': 0, 'deterministic': True, 'single_cls': False, 'rect': False, 'cos_lr': False, 'close_mosaic': 10, 'resume': False, 'resume_pt': None, 'amp': True, 'fraction': 1.0, 'profile': False, 'freeze': None, 'multi_scale': False, 'overlap_mask': True, 'mask_ratio': 4, 'dropout': 0.0, 'val': True, 'split': 'val', 'save_json': False, 'save_hybrid': False, 'conf': None, 'iou': 0.7, 'max_det': 300, 'half': False, 'dnn': False, 'plots': True, 'source': None, 'vid_stride': 1, 'stream_buffer': False, 'visualize': False, 'augment': False, 'agnostic_nms': False, 'classes': None, 'retina_masks': False, 'embed': None, 'show': False, 'save_frames': False, 'save_txt': False, 'save_conf': False, 'save_crop': False, 'show_labels': True, 'show_conf': True, 'show_boxes': True, 'line_width': None, 'format': 'torchscript', 'keras': False, 'optimize': False, 'int8': False, 'dynamic': False, 'simplify': False, 'opset': None, 'workspace': 4, 'nms': False, 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0, 'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'bgr': 0.0, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0, 'auto_augment': 'randaugment', 'erasing': 0.4, 'crop_fraction': 1.0, 'cfg': None, 'tracker': 'botsort.yaml', 'save_dir': 'runs/segment/train3'}

# if __name__ == "__main__":
#     from ultralytics.models.yolo.segment.train import SegmentationTrainer
#     from ultralytics.utils import DEFAULT_CFG_DICT

#     cfg = DEFAULT_CFG_DICT.copy()
#     cfg.update(save_dir='')   # handle the extra key 'save_dir'
#     trainer = SegmentationTrainer(cfg=cfg, overrides=overrides)
#     trainer.args.model_name = "yolov8l-seg.yaml"
#     results = trainer.train()
    
    
# ['/home/gcsx/anaconda3/envs/yolov8/bin/python', '-m', 'torch.distributed.run', '--nproc_per_node', '2', '--master_port', '40925', '/home/gcsx/.config/Ultralytics/DDP/_temp_fcvou9gf140280394578448.py']





# Ultralytics Multi-GPU training temp file (should be automatically deleted after use)
overrides = {'task': 'segment', 'mode': 'train', 'model_name': 'yolov8n-seg.yaml', 'data': 'ultralytics/cfg/datasets/coco128-seg.yaml', 'epochs': 3, 'time': None, 'patience': 100, 'batch': 2, 'imgsz': 640, 'save': True, 'save_period': -1, 'cache': False, 'device': '2', 'workers': 8, 'project': None, 'name': 'train2', 'exist_ok': False, 'pretrained': True, 'optimizer': 'auto', 'verbose': True, 'seed': 0, 'deterministic': True, 'single_cls': False, 'rect': False, 'cos_lr': False, 'close_mosaic': 10, 'resume': False, 'resume_pt': None, 'amp': True, 'fraction': 1.0, 'profile': False, 'freeze': None, 'multi_scale': False, 'overlap_mask': True, 'mask_ratio': 4, 'dropout': 0.0, 'val': True, 'split': 'val', 'save_json': False, 'save_hybrid': False, 'conf': None, 'iou': 0.7, 'max_det': 300, 'half': False, 'dnn': False, 'plots': True, 'source': None, 'vid_stride': 1, 'stream_buffer': False, 'visualize': False, 'augment': False, 'agnostic_nms': False, 'classes': None, 'retina_masks': False, 'embed': None, 'show': False, 'save_frames': False, 'save_txt': False, 'save_conf': False, 'save_crop': False, 'show_labels': True, 'show_conf': True, 'show_boxes': True, 'line_width': None, 'format': 'torchscript', 'keras': False, 'optimize': False, 'int8': False, 'dynamic': False, 'simplify': False, 'opset': None, 'workspace': 4, 'nms': False, 'lr0': 0.01, 'lrf': 0.01, 'momentum': 0.937, 'weight_decay': 0.0005, 'warmup_epochs': 3.0, 'warmup_momentum': 0.8, 'warmup_bias_lr': 0.1, 'box': 7.5, 'cls': 0.5, 'dfl': 1.5, 'pose': 12.0, 'kobj': 1.0, 'label_smoothing': 0.0, 'nbs': 64, 'hsv_h': 0.015, 'hsv_s': 0.7, 'hsv_v': 0.4, 'degrees': 0.0, 'translate': 0.1, 'scale': 0.5, 'shear': 0.0, 'perspective': 0.0, 'flipud': 0.0, 'fliplr': 0.5, 'bgr': 0.0, 'mosaic': 1.0, 'mixup': 0.0, 'copy_paste': 0.0, 'auto_augment': 'randaugment', 'erasing': 0.4, 'crop_fraction': 1.0, 'cfg': None, 'tracker': 'botsort.yaml', 'save_dir': 'runs/segment/train2'}

if __name__ == "__main__":
    from ultralytics.projects.yolo.segment.train import SegmentationTrainer
    from ultralytics.utils import DEFAULT_CFG_DICT

    cfg = DEFAULT_CFG_DICT.copy()
    cfg.update(save_dir='')   # handle the extra key 'save_dir'
    trainer = SegmentationTrainer(cfg=cfg, overrides=overrides)
    trainer.args.model_name = "yolov8n-seg.yaml"
    results = trainer.train()
    
    
# ['/home/gcsx/anaconda3/envs/yolov8/bin/python', '-m', 'torch.distributed.run', '--nproc_per_node', '2', '--master_port', '50061', '/home/gcsx/.config/Ultralytics/DDP/_temp_v3t6yewi140015092155856.py']