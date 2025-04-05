import os
import numpy as np

def expand_bbox(center_x, center_y, width, height, scale_factor, img_width, img_height):
    # 计算原始框的半宽和半高
    new_half_width = width * scale_factor / 2
    new_half_height = height * scale_factor / 2
    
    
    left = center_x - new_half_width
    right = center_x + new_half_width
    top = center_y - new_half_height
    bottom = center_y + new_half_height
    
    # 确保边界在0到1之间
    left = max(0, left)
    right = min(1, right)
    top = max(0, top)
    bottom = min(1, bottom)
    
    # 计算新的宽度和高度
    new_width = right - left
    new_height = bottom - top
    
    return center_x, center_y, new_width, new_height


def process_label_file(label_file, save_path, img_width, img_height, scale_factor=1.10):
    with open(label_file, 'r') as file:
        lines = file.readlines()
    
    new_labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        center_x, center_y, width, height = map(float, parts[1:])
        
        center_x, center_y, new_width, new_height = expand_bbox(center_x, center_y, width, height, scale_factor, img_width, img_height)
        
        new_labels.append(f"{class_id} {center_x} {center_y} {new_width} {new_height}")
    
    # 创建保存目录，如果不存在则创建
    os.makedirs(save_path, exist_ok=True)
    
    # 保存新的标签文件，使用原文件名
    new_file_name = os.path.basename(label_file)
    new_file_path = os.path.join(save_path, new_file_name)
    
    with open(new_file_path, 'w') as file:
        file.write("\n".join(new_labels))

def main(labels_path, save_path, img_width, img_height, scale_factor=1.10):
    label_files = [os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.txt')]
    
    for label_file in label_files:
        process_label_file(label_file, save_path, img_width, img_height, scale_factor)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Expand bounding boxes in label files.')
    parser.add_argument('--labels', default='Javeri-det-seg/valid_detect/labels', type=str, help='Path to the labeled images directory.')
    parser.add_argument('--save', default='Javeri-det-seg/valid_segment/MBB+20%/detect_labels', type=str, help='Path to save the expanded labels.')
    # parser.add_argument('--width', default=1, required=True, type=int, help='Width of the image.')
    # parser.add_argument('--height', default=1, required=True, type=int, help='Height of the image.')
    parser.add_argument('--scale', default=1.2, type=float, help='Scale factor for expanding bounding boxes.')
    
    args = parser.parse_args()
    
    main(args.labels, args.save, 1, 1, args.scale)