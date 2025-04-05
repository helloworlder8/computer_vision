import os
import numpy as np

# 假设 segments2boxes 函数已经定义
def segments2boxes(segments):
    # 将分割标签转换为矩形包围框的实现
    boxes = []
    for segment in segments:
        if len(segment) == 0:
            continue
        x_min = np.min(segment[:, 0])
        x_max = np.max(segment[:, 0])
        y_min = np.min(segment[:, 1])
        y_max = np.max(segment[:, 1])
        
        width = x_max - x_min
        height = y_max - y_min
        center_x = x_min + width / 2
        center_y = y_min + height / 2

        boxes.append([center_x, center_y, width, height])
    
    return np.array(boxes, dtype=np.float32)

def format_line(line):
    # Format the first element as integer and the rest as 6 decimal places
    return f"{int(line[0])} " + " ".join(f"{x:.6f}" for x in line[1:])

def process_labels(input_dir, output_dir, keypoint=False):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for file_name in os.listdir(input_dir):
        if file_name.endswith(".txt"):
            lb_file = os.path.join(input_dir, file_name)
            output_file = os.path.join(output_dir, file_name)
            
            if os.path.isfile(lb_file):
                nf = 1  # label found
                with open(lb_file) as f:
                    lb = [x.split() for x in f.read().strip().splitlines() if len(x)]
                    if any(len(x) > 6 for x in lb) and not keypoint:  # is segment
                        classes = np.array([x[0] for x in lb], dtype=np.float32)
                        segments = [np.array(x[1:], dtype=np.float32).reshape(-1, 2) for x in lb]  # (cls, xy1...)
                        lb = np.concatenate((classes.reshape(-1, 1), segments2boxes(segments)), 1)  # (cls, xywh)
                    lb = np.array(lb, dtype=np.float32)
                    
                    # Format and save to output file
                    with open(output_file, 'w') as out_file:
                        for line in lb:
                            formatted_line = format_line(line)
                            out_file.write(formatted_line + '\n')

# 设置输入输出目录
input_directory = 'select_image/2/'
output_directory = 'select_image/2/det_labels'

# 调用处理函数
process_labels(input_directory, output_directory)