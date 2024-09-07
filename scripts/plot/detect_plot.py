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

def annotated_image(image_folder, label_folder, output_folder, class_labels, class_colors_bgr):
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
            center_x = float(values[1]) * image_width
            center_y = float(values[2]) * image_height
            width = float(values[3]) * image_width
            height = float(values[4]) * image_height

            top_left_x = int(center_x - width / 2)
            top_left_y = int(center_y - height / 2)
            bottom_right_x = int(center_x + width / 2)
            bottom_right_y = int(center_y + height / 2)

            bbox_color = class_colors_bgr.get(class_id, (0, 0, 0))

            label = class_labels.get(class_id, 'unknown')

            cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), bbox_color, 2)
            
            cv2.putText(image, label, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

        output_image_path = os.path.join(output_folder, base_name + '_annotated.jpg')
        cv2.imwrite(output_image_path, image)

        print(f'Annotated image saved as {output_image_path}')

if __name__ == "__main__":
    image_folder = 'Javeri-det-seg/train_detect/images/'
    label_folder = 'Javeri-det-seg/train_detect/labels/'
    output_folder = 'valid_detect'

    class_labels = {0: 'human', 1: 'elephant', 2: 'giraffe', 3: 'unknown'}
    class_colors = {
        0: (255, 58, 56),    # human
        1: (253, 159, 150),  # elephant
        2: (254, 112, 35),   # giraffe
        3: (252, 173, 60)    # unknown

    }

    class_colors_bgr = {k: (v[2], v[1], v[0]) for k, v in class_colors.items()}
    
    annotated_image(image_folder, label_folder, output_folder, class_labels, class_colors_bgr)
