import os
import argparse
import numpy as np

def calculate_iou(bbox1, bbox2):
    # bbox format: [center_x, center_y, width, height]
    # Convert center format to top-left and bottom-right format
    x1_min = bbox1[0] - bbox1[2] / 2
    y1_min = bbox1[1] - bbox1[3] / 2
    x1_max = bbox1[0] + bbox1[2] / 2
    y1_max = bbox1[1] + bbox1[3] / 2

    x2_min = bbox2[0] - bbox2[2] / 2
    y2_min = bbox2[1] - bbox2[3] / 2
    x2_max = bbox2[0] + bbox2[2] / 2
    y2_max = bbox2[1] + bbox2[3] / 2

    # Calculate intersection coordinates
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # Calculate intersection area
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)

    # Calculate the area of each bounding box
    bbox1_area = (x1_max - x1_min) * (y1_max - y1_min)
    bbox2_area = (x2_max - x2_min) * (y2_max - y2_min)

    # Calculate union area
    union_area = bbox1_area + bbox2_area - inter_area

    # Calculate IoU
    iou = inter_area / union_area if union_area != 0 else 0.0
    return iou

def parse_detection_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    detections = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            print(f"Skipping invalid line (should have 5 elements): {line.strip()}")
            continue

        try:
            class_id = int(parts[0])
            bbox = list(map(float, parts[1:5]))  # center_x, center_y, width, height
            detections.append((class_id, bbox))
        except ValueError as e:
            print(f"Skipping invalid line (conversion error): {line.strip()}, error: {e}")
            continue

    return detections

def main(labels_path, pred_path, per_class):
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('.txt')]

    iou_results_file = os.path.join(pred_path, '00_detection_iou_results.txt')
    with open(iou_results_file, 'w') as f:
        valid_total_iou = 0.0
        valid_iou_count = 0  
        low_iou_count = 0    

        for label_file in label_files: #每一个文件
            label_data = parse_detection_file(os.path.join(labels_path, label_file))
            pred_data = parse_detection_file(os.path.join(pred_path, label_file))

            image_iou_sum = 0.0
            image_iou_count = 0

            for label_class, label_bbox in label_data:
                for pred_class, pred_bbox in pred_data:
                    if label_class == pred_class: #类别相等的情况下
                        iou = calculate_iou(label_bbox, pred_bbox)
                        if iou > 0.7:  # Only consider IoU > 0.9
                            if per_class: #逐类别计算
                                f.write(f"{label_file} class {label_class}: {iou}\n")
                                print(f"{label_file} class {label_class}: {iou}")
                            else:
                                image_iou_sum += iou
                                image_iou_count += 1
                            valid_total_iou += iou
                            valid_iou_count += 1
                        else:
                            low_iou_count += 1

            if not per_class and image_iou_count > 0:  # 逐图像
                image_avg_iou = image_iou_sum / image_iou_count
                f.write(f"{label_file} average IoU: {image_avg_iou}\n") 
                print(f"{label_file} average IoU: {image_avg_iou}")


        valid_mean_iou = valid_total_iou / valid_iou_count if valid_iou_count > 0 else 0.0
        f.write(f"valid Mean IoU (per class): {valid_mean_iou}\n")
        print(f"valid Mean IoU (per class): {valid_mean_iou}")


        f.write(f"Number of IoUs <= 0.9: {low_iou_count}\n") #数目
        print(f"Number of IoUs <= 0.9: {low_iou_count}")

        # Compute the ratio of IoU > 0.9 for overall evaluation
        ratio = valid_iou_count / (valid_iou_count + low_iou_count) if (valid_iou_count + low_iou_count) > 0 else 0
        f.write(f"比例: {ratio}\n")
        print(f"比例: {ratio}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute IoU for object detection results.')
    parser.add_argument('--labels', default='val/labels', type=str, help='Path to the ground truth labels directory.')
    parser.add_argument('--pred', default='val/label_yolo11x', type=str, help='Path to the predicted labels directory.')
    parser.add_argument('--per_class', action='store_true', help='Calculate IoU for each individual label rather than per image.')
    args = parser.parse_args()

    main(args.labels, args.pred, args.per_class)
