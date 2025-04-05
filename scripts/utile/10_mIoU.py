import os
import argparse
import numpy as np
import cv2

def points_to_mask(points, img_shape):
    # Check if img_shape is valid
    if len(img_shape) < 2:
        print("Invalid image shape:", img_shape)
        return None

    mask = np.zeros(img_shape, dtype=np.uint8)
    img_height, img_width = img_shape
    points = np.array(points).reshape(-1, 2)  # Ensure points are reshaped to (N, 2)

    # Scale points to the image dimensions
    points = (points * [img_width, img_height]).astype(np.int32)

    try:
        cv2.fillPoly(mask, [points], 1)
    except Exception as e:
        print(f"Error filling polygon with points: {points}, error: {e}")
        return None

    return mask

def accumulate_masks(mask_dict, class_id, points, img_shape):
    new_mask = points_to_mask(points, img_shape)
    if new_mask is None:
        print(f"Skipping accumulation due to mask generation error for class {class_id}.")
        return

    if class_id in mask_dict:
        mask_dict[class_id] = np.logical_or(mask_dict[class_id], new_mask).astype(np.uint8)
    else:
        mask_dict[class_id] = new_mask

def parse_label_file(file_path):
    try:
        with open(file_path, 'r') as file:
            lines = file.readlines()
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return []

    labels = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) < 7:
            print(f"Skipping invalid line (less than 7 elements): {line.strip()}")
            continue
        
        try:
            class_id = int(parts[0])
            points = list(map(float, parts[1:]))
            if len(points) % 2 != 0:
                print(f"Skipping invalid line (odd number of coordinates): {line.strip()}")
                continue
            points = [[points[i], points[i+1]] for i in range(0, len(points), 2)]
            labels.append((class_id, points))
        except ValueError as e:
            print(f"Skipping invalid line (conversion error): {line.strip()}, error: {e}")
            continue

    return labels

def compute_iou(label_mask, pred_mask):
    intersection = np.logical_and(label_mask, pred_mask)
    union = np.logical_or(label_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0.0
    return iou

def main(labels_path, pred_path, images_path):
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('.txt')]

    iou_results_file = os.path.join(pred_path, '1_iou_results.txt')
    with open(iou_results_file, 'w') as f:
        total_iou = 0.0
        valid_iou_count = 0  # Count of IoU values > 0.75
        low_iou_count = 0    # Count of IoU values <= 0.75

        for label_file in label_files:
            img_file = label_file.replace('.txt', '.jpg')  # Assuming the image file is JPG
            img_path = os.path.join(images_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading image {img_path}")
                continue

            img_shape = img.shape[:2]  # (height, width)
            print(f"Processing {label_file}, image shape: {img_shape}")

            label_data = parse_label_file(os.path.join(labels_path, label_file))
            pred_data = parse_label_file(os.path.join(pred_path, label_file))

            label_masks = {}
            for class_id, points in label_data:
                accumulate_masks(label_masks, class_id, points, img_shape)

            pred_masks = {}
            for class_id, points in pred_data:
                accumulate_masks(pred_masks, class_id, points, img_shape)

            for class_id in label_masks:
                if class_id in pred_masks:
                    iou = compute_iou(label_masks[class_id], pred_masks[class_id])
                    if iou > 0.75:  # Only consider IoU > 0.75
                        total_iou += iou
                        valid_iou_count += 1
                    else:
                        low_iou_count += 1
                    f.write(f"{label_file} class {class_id}: {iou}\n")
                    print(f"{label_file} class {class_id}: {iou}")

        mean_iou = total_iou / valid_iou_count if valid_iou_count > 0 else 0.0
        f.write(f"Mean IoU: {mean_iou}\n")
        print(f"Mean IoU: {mean_iou}")

        f.write(f"Number of IoUs <= 0.75: {low_iou_count}\n")
        print(f"Number of IoUs <= 0.75: {low_iou_count}")

        f.write(f"比例: {valid_iou_count/(valid_iou_count+low_iou_count)}\n")
        print(f"比例: {valid_iou_count/(valid_iou_count+low_iou_count)}")
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute mIoU for labeled and predicted images.')
    parser.add_argument('--labels', default='ADE20K_2016_yolo/valid_segment/labels', type=str, help='Path to the labeled images directory.')
    parser.add_argument('--pred', default='ADE20K_2016_yolo/valid_segment/5_points/mobile_sam_labels', type=str, help='Path to the predicted images directory.')
    parser.add_argument('--images', default='ADE20K_2016_yolo/valid_segment/images', type=str, help='Path to the images directory.')
    args = parser.parse_args()

    main(args.labels, args.pred, args.images)


    parser = argparse.ArgumentParser(description='Compute mIoU for labeled and predicted images.')
    parser.add_argument('--labels', default='ADE20K_2016_yolo/valid_segment/labels', type=str, help='Path to the labeled images directory.')
    parser.add_argument('--pred', default='ADE20K_2016_yolo/valid_segment/5_points/sam_b_labels', type=str, help='Path to the predicted images directory.')
    parser.add_argument('--images', default='ADE20K_2016_yolo/valid_segment/images', type=str, help='Path to the images directory.')
    args = parser.parse_args()

    main(args.labels, args.pred, args.images)
    
    parser = argparse.ArgumentParser(description='Compute mIoU for labeled and predicted images.')
    parser.add_argument('--labels', default='ADE20K_2016_yolo/valid_segment/labels', type=str, help='Path to the labeled images directory.')
    parser.add_argument('--pred', default='ADE20K_2016_yolo/valid_segment/5_points/sam_l_labels', type=str, help='Path to the predicted images directory.')
    parser.add_argument('--images', default='ADE20K_2016_yolo/valid_segment/images', type=str, help='Path to the images directory.')
    args = parser.parse_args()

    main(args.labels, args.pred, args.images)

    parser = argparse.ArgumentParser(description='Compute mIoU for labeled and predicted images.')
    parser.add_argument('--labels', default='ADE20K_2016_yolo/valid_segment/labels', type=str, help='Path to the labeled images directory.')
    parser.add_argument('--pred', default='ADE20K_2016_yolo/valid_segment/5_points/sam2_t_labels', type=str, help='Path to the predicted images directory.')
    parser.add_argument('--images', default='ADE20K_2016_yolo/valid_segment/images', type=str, help='Path to the images directory.')
    args = parser.parse_args()

    main(args.labels, args.pred, args.images)
    
    
    parser = argparse.ArgumentParser(description='Compute mIoU for labeled and predicted images.')
    parser.add_argument('--labels', default='ADE20K_2016_yolo/valid_segment/labels', type=str, help='Path to the labeled images directory.')
    parser.add_argument('--pred', default='ADE20K_2016_yolo/valid_segment/5_points/sam2_s_labels', type=str, help='Path to the predicted images directory.')
    parser.add_argument('--images', default='ADE20K_2016_yolo/valid_segment/images', type=str, help='Path to the images directory.')
    args = parser.parse_args()

    main(args.labels, args.pred, args.images)
    
    parser = argparse.ArgumentParser(description='Compute mIoU for labeled and predicted images.')
    parser.add_argument('--labels', default='ADE20K_2016_yolo/valid_segment/labels', type=str, help='Path to the labeled images directory.')
    parser.add_argument('--pred', default='ADE20K_2016_yolo/valid_segment/5_points/sam2_b_labels', type=str, help='Path to the predicted images directory.')
    parser.add_argument('--images', default='ADE20K_2016_yolo/valid_segment/images', type=str, help='Path to the images directory.')
    args = parser.parse_args()

    main(args.labels, args.pred, args.images)
    
    parser = argparse.ArgumentParser(description='Compute mIoU for labeled and predicted images.')
    parser.add_argument('--labels', default='ADE20K_2016_yolo/valid_segment/labels', type=str, help='Path to the labeled images directory.')
    parser.add_argument('--pred', default='ADE20K_2016_yolo/valid_segment/5_points/sam2_l_labels', type=str, help='Path to the predicted images directory.')
    parser.add_argument('--images', default='ADE20K_2016_yolo/valid_segment/images', type=str, help='Path to the images directory.')
    args = parser.parse_args()

    main(args.labels, args.pred, args.images)
