# 定义文件路径列表
file_paths = [
    'result/Ablation_experiment_exp_val/M5_con0.3_labels/1_iou_results.txt',
    # 'ADE20K_2016_yolo/valid_segment/MMB+20%/sam_b_labels/1_iou_results.txt',
    # 'ADE20K_2016_yolo/valid_segment/MMB+20%/sam_l_labels/1_iou_results.txt',
    # 'ADE20K_2016_yolo/valid_segment/MMB+20%/sam2_t_labels/1_iou_results.txt',
    # 'ADE20K_2016_yolo/valid_segment/MMB+20%/sam2_s_labels/1_iou_results.txt',
    # 'ADE20K_2016_yolo/valid_segment/MMB+20%/sam2_b_labels/1_iou_results.txt',
    # 'ADE20K_2016_yolo/valid_segment/MMB+20%/sam2_l_labels/1_iou_results.txt',
]

# 定义计算平均IoU的函数
def calculate_average_iou(file_path):
    total_iou = 0
    count = 0

    with open(file_path, 'r') as file:
        lines = file.readlines()

        # 排除最后三行
        for line in lines[:-3]:
            parts = line.strip().split(':')
            if len(parts) == 2:
                iou_value = float(parts[1].strip())
                total_iou += iou_value
                count += 1

    average_iou = total_iou / count if count > 0 else 0
    return average_iou

# 循环处理每个文件并打印平均IoU值
for file_path in file_paths:
    avg_iou = calculate_average_iou(file_path)
    print(f'{file_path} 的平均IoU值: {avg_iou}')
