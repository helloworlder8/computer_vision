import cv2
import os
import glob
import numpy as np

# Function to generate random colors
def random_color():
    return tuple(np.random.randint(0, 256, size=3).tolist())

class Colors:
    """
    Ultralytics default color palette https://ultralytics.com/.

    This class provides methods to work with the Ultralytics color palette, including converting hex color codes to
    RGB values.

    Attributes:
        palette (list of tuple): List of RGB color values.
        n (int): The number of colors in the palette.
        pose_palette (np.ndarray): A specific color palette array with dtype np.uint8.
    """
# 4d9c97
    def __init__(self):
        """Initialize colors as hex = matplotlib.colors.TABLEAU_COLORS.values()."""
        hexs = (
            "042AFF",
            "0BDBEB",
            "F3F3F3",
            "00DFB7",
            "111F68",
            "FF6FDD",
            "FF444F",
            "CCED00",
            "00F344",
            "BD00FF",
            "00B4FF",
            "DD00BA",
            "00FFFF",
            "26C000",
            "01FFB3",
            "7D24FF",
            "7B0068",
            "FF1B6C",
            "FC6D2F",
            "A2FF0B",
        )
        self.palette = [self.hex2rgb(f"#{c}") for c in hexs]
        self.n = len(self.palette)
        self.pose_palette = np.array(
            [
                [255, 128, 0],
                [255, 153, 51],
                [255, 178, 102],
                [230, 230, 0],
                [255, 153, 255],
                [153, 204, 255],
                [255, 102, 255],
                [255, 51, 255],
                [102, 178, 255],
                [51, 153, 255],
                [255, 153, 153],
                [255, 102, 102],
                [255, 51, 51],
                [153, 255, 153],
                [102, 255, 102],
                [51, 255, 51],
                [0, 255, 0],
                [0, 0, 255],
                [255, 0, 0],
                [255, 255, 255],
            ],
            dtype=np.uint8,
        )

    def __call__(self, i, bgr=False):
        """Converts hex color codes to RGB values."""
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        """Converts hex color codes to RGB values (i.e. default PIL order)."""
        return tuple(int(h[1 + i : 1 + i + 2], 16) for i in (0, 2, 4))



# class_names = ['person', 'garbage', 'pit', 'trouble',
#                 'sink', 'stone', 'crack_p', 'patch_crack_p', 'crack_t', 
#                 'patch_crack_t', 'net', 'patch_net', 'swell',  'rut', 
#                 'stretch', 'drainage', 'manhole', 'guardrail', 'deadened',  
#                 'column', 'bucket',  'screen',  'flash',  'frame', 
#                 'brige',  'barrier',  'mirror',  'pier', 'advert',  
#                 'crash',  'manhole_d', 'guardrail_d',  'stretch_d', 'drainage_d', 
#                 'rockfall', 'barrier_d', 'column_d', 'stone_d']
# class_names = (
#     'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
#     'fire hydrant',
#     'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
#     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
#     'kite',
#     'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork',
#     'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
#     'donut',
#     'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
#     'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase',
#     'scissors',
#     'teddy bear', 'hair drier', 'toothbrush')
class_names = ["D00", "D01", "D03", "D03", "D04", "D05", "D06", "D07"]

custom_colors_bgr = [(106,100,179), (79, 68, 255), (221,189,103), (0,0,103),
                  (186, 0, 221), (0, 192, 38),  (96,64,121), (104, 0, 123),]

custom_colors_rgb = [(r, g, b) for (b, g, r) in custom_colors_bgr]
    
# Function to draw custom markers at the center points
def draw_custom_marker(image, center, color, marker_type, size):
    x, y = center
    if marker_type == '*':
        # Draw a star shape
        cv2.drawMarker(image, (x, y), color, markerType=cv2.MARKER_STAR, markerSize=size, thickness=1, line_type=cv2.LINE_AA)
    elif marker_type == 'o':
        # Draw a circle with edge color
        cv2.circle(image, (x, y), size // 2, color, thickness=-1)  # Filled circle
        cv2.circle(image, (x, y), size // 2, (255, 255, 255), thickness=1)  # Edge circle
    else:
        # Default to a cross if the marker type is not recognized
        cv2.drawMarker(image, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=1, line_type=cv2.LINE_AA)

# Main function to draw bounding boxes and optional center points with customization
def annotated_image_detection(
    image_folder, label_folder, output_folder, file_extension="JPG", color_mode=1, draw_center_points=False, marker_type='*', marker_size=10, 
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(image_folder, f'*.{file_extension}'))
    color_palette = Colors()

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

        num = 1
        Instance =0
        for line in lines:
            values = line.split()
            class_id = int(values[0])  # Extract the class ID
            x_center = float(values[1]) * image_width
            y_center = float(values[2]) * image_height
            box_width = float(values[3]) * image_width
            box_height = float(values[4]) * image_height

            # Calculate the top-left and bottom-right corners of the bounding box
            x1 = max(0, int(x_center - box_width / 2))
            y1 = max(0, int(y_center - box_height / 2))
            x2 = min(int(x_center + box_width / 2),image_width)
            y2 = min(int(y_center + box_height / 2),image_height)

            # Choose the color for the bounding box
            if color_mode==0:
                color = random_color()
            elif color_mode==1:
                color = color_palette(class_id, True)
            elif color_mode==2:
                num += 1
                color = color_palette(num, True)
            else:
                color = custom_colors_rgb[class_id]
                
            # if class_id == 68 or class_id == 109 or class_id == 72:
            if True:
                p1= (x1,y1)
                p2= (x2,y2)
                
                p1, p2 = (int(x1), int(y1)), (int(x2), int(y2))
                cv2.rectangle(image, p1, p2, color, thickness=3, lineType=cv2.LINE_AA)
                
                
                w, h = cv2.getTextSize(class_names[class_id], 0, fontScale=1, thickness=2)[0]  # text width, height
                h += 3  # add pixels to pad text
                outside = p1[1] >= h  # label fits outside box
                if p1[0] > image.shape[1] - w:  # shape is (h, w), check if label extend beyond right side of image
                    p1 = image.shape[1] - w, p1[1]
                p2 = p1[0] + w, p1[1] - h if outside else p1[1] + h
                # cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # 原始图像 两个边角的坐标 颜色板（list）被填充默认值 cv2.LINE_AA 是抗锯齿线型
                
                text_name = f"{Instance}-{class_names[class_id]}"
                cv2.putText(
                    image,            # 原始图像
                    text_name,               # 标签字符串
                    (p1[0], p1[1] - 2 if outside else p1[1] + h - 1),  # 文字起始位置
                    0,                   # 字体类型（这里传入 0，表示默认字体）
                    1,             # 字体缩放因子
                    (255,255,255),           # 文字颜色
                    thickness=2,   # 文字线条粗细
                    lineType=cv2.LINE_AA # 线型（抗锯齿）
                )
                Instance +=1

                # Draw the center point with custom style if the option is enabled
                if draw_center_points:
                     draw_custom_marker(image, (int(x_center), int(y_center)), color, marker_type, marker_size)

        output_image_path = os.path.join(output_folder, base_name + '_annotatedv2.jpg')
        cv2.imwrite(output_image_path, image)
        print(f'Annotated image saved as {output_image_path}')



if __name__ == "__main__":
    # Parameters can be modified as needed
    image_folder = 'images'
    label_folder = 'label_txt'
    output_folder = 'vision/'
    file_extension = "jpg"  # Image file extension
    color_mode = 1  # Set to True if you want to use random colors
    

    # Custom marker options
    draw_center_points = False  # Set to True to draw center points
    marker_type = '*'  # Supported types: '*', 'o', etc.
    marker_size = 2  # Marker size



    annotated_image_detection(
        image_folder=image_folder,
        label_folder=label_folder,
        output_folder=output_folder,
        file_extension=file_extension,
        color_mode=color_mode,
        draw_center_points=draw_center_points,
        marker_type=marker_type,
        marker_size=marker_size,


    )