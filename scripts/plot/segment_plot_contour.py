import cv2
import os
import glob
import numpy as np

class Colors:
    def __init__(self):
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
        c = self.palette[int(i) % self.n]
        return (c[2], c[1], c[0]) if bgr else c

    @staticmethod
    def hex2rgb(h):
        return tuple(int(h[1 + i: 1 + i + 2], 16) for i in (0, 2, 4))

colors = Colors()

# Sample class labels (You should replace these with your actual class labels)
class_labels = {
    0: 'person',
    1: 'bicycle',
    2: 'car',
    3: 'motorcycle',
    4: 'airplane',
    5: 'bus',
    6: 'train',
    7: 'truck',
    8: 'boat',
    9: 'traffic light',
    10: 'fire hydrant',
    11: 'stop sign',
    12: 'parking meter',
    13: 'bench',
    14: 'bird',
    15: 'cat',
    16: 'dog',
    17: 'horse',
    18: 'sheep',
    19: 'cow',
    20: 'elephant',
    21: 'bear',
    22: 'zebra',
    23: 'giraffe',
    24: 'backpack',
    25: 'umbrella',
    26: 'handbag',
    27: 'tie',
    28: 'suitcase',
    29: 'frisbee',
    30: 'skis',
    31: 'snowboard',
    32: 'sports ball',
    33: 'kite',
    34: 'baseball bat',
    35: 'baseball glove',
    36: 'skateboard',
    37: 'surfboard',
    38: 'tennis racket',
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    52: 'hot dog',
    53: 'pizza',
    54: 'donut',
    55: 'cake',
    56: 'chair',
    57: 'couch',
    58: 'potted plant',
    59: 'bed',
    60: 'dining table',
    61: 'toilet',
    62: 'tv',
    63: 'laptop',
    64: 'mouse',
    65: 'remote',
    66: 'keyboard',
    67: 'cell phone',
    68: 'microwave',
    69: 'oven',
    70: 'toaster',
    71: 'sink',
    72: 'refrigerator',
    73: 'book',
    74: 'clock',
    75: 'vase',
    76: 'scissors',
    77: 'teddy bear',
    78: 'hair drier',
    79: 'toothbrush'
}


# Sample class colors (You can use Colors class or define your own)
class_colors_bgr = {i: colors(i, bgr=True) for i in class_labels.keys()}

def annotated_image_segmentation(image_folder, label_folder, output_folder, class_labels, class_colors_bgr):
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
            
            class_id = int(values[0])
            points = np.array([float(v) for v in values[1:]]).reshape(-1, 2)
            points[:, 0] *= image_width
            points[:, 1] *= image_height
            points = points.astype(np.int32)

            polygon_color = class_colors_bgr.get(class_id, (0, 0, 0))

            label = class_labels.get(class_id, 'unknown')

            cv2.polylines(image, [points], isClosed=True, color=polygon_color, thickness=2)
            
            centroid_x = int(np.mean(points[:, 0]))
            centroid_y = int(np.mean(points[:, 1]))

            cv2.putText(image, label, (centroid_x, centroid_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, polygon_color, 2)

        output_image_path = os.path.join(output_folder, base_name + '_annotated.jpg')
        cv2.imwrite(output_image_path, image)

        print(f'Annotated image saved as {output_image_path}')

if __name__ == "__main__":
    image_folder = 'coco_yolo/valid/images/'
    label_folder = 'coco_yolo/valid/labels/'
    output_folder = 'valid_segment'

    annotated_image_segmentation(image_folder, label_folder, output_folder, class_labels, class_colors_bgr)
