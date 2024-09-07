import os
import argparse
import numpy as np
import cv2

def points_to_mask(points, img_shape):
    mask = np.zeros(img_shape, dtype=np.uint8)
    
    # 如果 points 是相对坐标，需要乘以图像的宽度和高度
    img_height, img_width = img_shape
    points = np.array(points).reshape(-1, 2)  # (N, 2)
    points = (points * [img_width, img_height]).astype(np.int32)
    
    cv2.fillPoly(mask, [points], 1)
    return mask

def accumulate_masks(mask_dict, class_id, points, img_shape):
    new_mask = points_to_mask(points, img_shape)
    if class_id in mask_dict:
        mask_dict[class_id] = np.logical_or(mask_dict[class_id], new_mask).astype(np.uint8)
    else:
        mask_dict[class_id] = new_mask
        
def parse_label_file(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    labels = []
    for line in lines:
        parts = line.strip().split()
        class_id = int(parts[0])
        points = list(map(float, parts[1:]))
        points = [[points[i], points[i+1]] for i in range(0, len(points), 2)]
        labels.append((class_id, points))
    return labels

def compute_iou(label_mask, pred_mask):
    intersection = np.logical_and(label_mask, pred_mask)
    union = np.logical_or(label_mask, pred_mask)
    iou = np.sum(intersection) / np.sum(union)
    return iou

def main(labels_path, pred_path, images_path):
    label_files = [f for f in os.listdir(labels_path) if f.endswith('.txt')]
    pred_files = [f for f in os.listdir(pred_path) if f.endswith('.txt')]

    if not os.path.exists('results'):
        os.makedirs('results')

    with open('results/iou_results.txt', 'w') as f:
        total_iou = 0.0
        for label_file in label_files:
                        # Load image to get its shape
            img_file = label_file.replace('.txt', '.jpg')  # Assuming the image file is PNG
            img_path = os.path.join(images_path, img_file)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Error loading image {img_path}")
                continue
            img_shape = img.shape[:2]  # (height, width)
            
            label_data = parse_label_file(os.path.join(labels_path, label_file))
            pred_data = parse_label_file(os.path.join(pred_path, label_file))

            # img_shape = (1080, 1920)  # Example image shape, you can change it as needed
            
            label_masks = {}
            for class_id, points in label_data:
                accumulate_masks(label_masks, class_id, points, img_shape)
            
            pred_masks = {}
            for class_id, points in pred_data:
                accumulate_masks(pred_masks, class_id, points, img_shape)

            for class_id in label_masks:
                if class_id in pred_masks:
                    iou = compute_iou(label_masks[class_id], pred_masks[class_id])
                    total_iou += iou
                    f.write(f"{label_file} class {class_id}: {iou}\n")
                    print(f"{label_file} class {class_id}: {iou}")

        mean_iou = total_iou / len(label_files)
        f.write(f"Mean IoU: {mean_iou}\n")
        print(f"Mean IoU: {mean_iou}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute mIoU for labeled and predicted images.')
    parser.add_argument('--labels', required=True, type=str, help='Path to the labeled images directory.')
    parser.add_argument('--pred', required=True, type=str, help='Path to the predicted images directory.')
    parser.add_argument('--images', required=True, type=str, help='Path to the images directory.')
    args = parser.parse_args()

    main(args.labels, args.pred, args.images)
