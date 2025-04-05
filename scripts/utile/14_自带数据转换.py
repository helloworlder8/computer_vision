# from ultralytics.data.converter import convert_coco

# convert_coco("../datasets/coco1/", use_segments=True, use_keypoints=False, cls91to80=False)


from ultralytics.data.converter import convert_segment_masks_to_yolo_seg

# The classes here is the total classes in the dataset, for COCO dataset we have 80 classes
convert_segment_masks_to_yolo_seg("/media/ang/2T/datasets/未处理的数据集/ADE20K_2016/annotations/train", "./ADE20K_2016/", classes=150)