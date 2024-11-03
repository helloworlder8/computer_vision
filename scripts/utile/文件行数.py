file_path = 'ADE20K_2016_yolo/valid_segment/MMB/sam_l_labels/1_iou_results.txt'

# 统计文件总行数
with open(file_path, 'r') as file:
    total_lines = len(file.readlines())

print(f"Total number of lines: {total_lines}")
