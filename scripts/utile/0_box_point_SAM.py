# from ultralytics.data.converter import yolo_points2segment
from ultralytics.data.converter import yolo_points2segment

yolo_points2segment(  
    im_dir="../datasets/ADE20K_2016_yolo/valid_detect/images",
    save_dir="../datasets/ADE20K_2016_yolo/valid_segment/1_point/mobile_sam_labels",  # saved to "labels-segment" in images directory
    sam_model="../checkpoints/mobile_sam.pt",
)
# ADE20K_2016_yolo/valid_detect/images

yolo_points2segment(  
    im_dir="../datasets/ADE20K_2016_yolo/valid_detect/images",
    save_dir="../datasets/ADE20K_2016_yolo/valid_segment/1_point/sam_b_labels",  # saved to "labels-segment" in images directory
    sam_model="../checkpoints/sam_b.pt",
)


yolo_points2segment(  
    im_dir="../datasets/ADE20K_2016_yolo/valid_detect/images",
    save_dir="../datasets/ADE20K_2016_yolo/valid_segment/1_point/sam_l_labels",  # saved to "labels-segment" in images directory
    sam_model="../checkpoints/sam_l.pt",
)


yolo_points2segment(  
    im_dir="../datasets/ADE20K_2016_yolo/valid_detect/images",
    save_dir="../datasets/ADE20K_2016_yolo/valid_segment/1_point/sam2_t_labels",  # saved to "labels-segment" in images directory
    sam_model="../checkpoints/sam2_t.pt",
)

yolo_points2segment(  
    im_dir="../datasets/ADE20K_2016_yolo/valid_detect/images",
    save_dir="../datasets/ADE20K_2016_yolo/valid_segment/1_point/sam2_s_labels",  # saved to "labels-segment" in images directory
    sam_model="../checkpoints/sam2_s.pt",
)

yolo_points2segment(  
    im_dir="../datasets/ADE20K_2016_yolo/valid_detect/images",
    save_dir="../datasets/ADE20K_2016_yolo/valid_segment/1_point/sam2_b_labels",  # saved to "labels-segment" in images directory
    sam_model="../checkpoints/sam2_b.pt",
)

yolo_points2segment(  
    im_dir="../datasets/ADE20K_2016_yolo/valid_detect/images",
    save_dir="../datasets/ADE20K_2016_yolo/valid_segment/1_point/sam2_l_labels",  # saved to "labels-segment" in images directory
    sam_model="../checkpoints/sam2_l.pt",
)



# sam_model_map = { #一系列内存空间
#     "sam_h.pt": build_sam_vit_h, 这个没有官方权重
#     "sam_l.pt": build_sam_vit_l,
#     "sam_b.pt": build_sam_vit_b,
#     "mobile_sam.pt": build_mobile_sam,
#     "sam2_t.pt": build_sam2_t,
#     "sam2_s.pt": build_sam2_s,
#     "sam2_b.pt": build_sam2_b,
#     "sam2_l.pt": build_sam2_l,
# }