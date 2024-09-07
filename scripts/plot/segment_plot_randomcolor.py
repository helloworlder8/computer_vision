import cv2
import os
import glob
import numpy as np

# 使用随机颜色生成函数
def random_color():
    return tuple(np.random.randint(0, 256, size=3).tolist())



def draw_filled_polygon_with_transparency(image, points, label, polygon_color, alpha=0):
    # 创建与原始图像相同大小的蒙版
    overlay = image.copy()
    
    # 绘制填充的多边形在蒙版上
    cv2.fillPoly(overlay, [points], color=polygon_color)

    # 将蒙版与原始图像进行透明度融合
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)
    
    # 选择多边形的第一个点作为标签的放置位置
    label_position = (points[0][0], points[0][1] - 10)  # 标签显示在第一个顶点的上方
    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

                
                
def annotated_image_segmentation(image_folder, label_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(image_folder, '*.jpg'))

    for image_path in image_files:
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotation_path = os.path.join(label_folder, base_name + '.txt')
        
        if not os.path.exists(annotation_path):
            print(f"Annotation file {annotation_path} not found for image {image_path}. Skipping.")
            continue

        image = cv2.imread(image_path)

        with open(annotation_path, 'r') as file:
            lines = file.readlines()

        image_height, image_width = image.shape[:2]

        for line in lines:
            values = line.split()
            label = values[0]
            if label == '129':
                points = np.array([float(v) for v in values[1:]]).reshape(-1, 2)  # 忽略类ID，只取多边形坐标
                points[:, 0] *= image_width
                points[:, 1] *= image_height
                points = points.astype(np.int32)

                polygon_color = random_color()  # 使用随机颜色

                # 绘制多边形
                cv2.polylines(image, [points], isClosed=True, color=polygon_color, thickness=2)
                # 选择多边形的第一个点作为标签的放置位置
                label_position = (points[0][0], points[0][1] - 10)  # 标签显示在第一个顶点的上方
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, polygon_color, 2)

                if True:
                    draw_filled_polygon_with_transparency(image, points, label, polygon_color, alpha=0.5)
                else:
                                        # 绘制多边形
                    cv2.polylines(image, [points], isClosed=True, color=polygon_color, thickness=2)
                    # 选择多边形的第一个点作为标签的放置位置
                    label_position = (points[0][0], points[0][1] - 10)  # 标签显示在第一个顶点的上方
                    cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, polygon_color, 2)

                
        output_image_path = os.path.join(output_folder, base_name + '_annotated.jpg')
        cv2.imwrite(output_image_path, image)

        print(f'Annotated image saved as {output_image_path}')

if __name__ == "__main__":
    image_folder = './'
    label_folder = './'
    output_folder = 'gt_segment'

    annotated_image_segmentation(image_folder, label_folder, output_folder)
