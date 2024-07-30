import cv2

def resize_image(image_path, output_path, max_size=1080):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to read image from {image_path}")
        return

    # 获取图像的尺寸
    height, width = image.shape[:2]

    # 计算缩放比例
    if max(height, width) > max_size:
        scale = max_size / max(height, width)
    else:
        scale = 1

    # 计算新尺寸
    new_width = int(width * scale)
    new_height = int(height * scale)

    # 缩放图像
    resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    # 保存缩放后的图像
    cv2.imwrite(output_path, resized_image)
    print(f"Resized image saved to {output_path}")

if __name__ == "__main__":
    input_image_path = 'DJI_20240323135637_0023_D.JPG'
    output_image_path = 'DJI_20240323135637_0023_D_resized.JPG'
    resize_image(input_image_path, output_image_path)
