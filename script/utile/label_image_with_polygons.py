import cv2
import numpy as np

# 读取并解析标签数据文件
def read_and_parse_label_file(file_path):
    labels = []
    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            parts = list(map(float, line.split()))
            label = int(parts[0])
            coordinates = np.array(parts[1:]).reshape(-1, 2)
            labels.append((label, coordinates))
    return labels

# 将归一化坐标转换为图像坐标
def denormalize_coordinates(coords, width, height):
    coords[:, 0] *= width
    coords[:, 1] *= height
    return coords.astype(int)

def main():
    image_path = 'ultralytics/assets/zidane.jpg'
    label_file_path = '1.txt'
    
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image from {image_path}")
        return

    h, w, _ = image.shape

    # 读取并解析标签数据
    labels = read_and_parse_label_file(label_file_path)

    # 绘制多边形
    for label, coords in labels:
        coords = denormalize_coordinates(coords, w, h)
        cv2.polylines(image, [coords], isClosed=True, color=(0, 255, 0), thickness=2)

    output_path = '000000000007_labeled.jpg'
    cv2.imwrite(output_path, image)
    print(f"Labeled image saved to {output_path}")

if __name__ == "__main__":
    main()