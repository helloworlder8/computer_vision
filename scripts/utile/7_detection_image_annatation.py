import cv2
import os
import glob
import numpy as np

class Colors:

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

# (253,159,150) 1 (255,58,56) 0 (252,173,60) 3 (254,112,35) 2
colors = Colors()  # create instance for 'from utils.plots import colors'




def annotated_image(directory_path,class_labels,class_colors_bgr):

    # Get a list of all .jpg files in the directory
    image_files = glob.glob(os.path.join(directory_path, '*.jpg'))

    for image_path in image_files:
        # Get the base name (without extension) to find the corresponding .txt file
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        annotation_path = os.path.join(directory_path, base_name + '.txt')
        
        # Check if the corresponding .txt file exists
        if not os.path.exists(annotation_path):
            print(f"Annotation file {annotation_path} not found for image {image_path}. Skipping.")
            continue

        # Load the image
        image = cv2.imread(image_path)

        # Read the annotation file
        with open(annotation_path, 'r') as file:
            lines = file.readlines()

        # Get image dimensions
        image_height, image_width = image.shape[:2]

        # Parse each line in the annotation file
        for line in lines:
            values = line.split()
            
            # Extract class and bounding box details
            class_id = int(values[0])
            center_x = float(values[1]) * image_width
            center_y = float(values[2]) * image_height
            width = float(values[3]) * image_width
            height = float(values[4]) * image_height

            # Calculate the top-left and bottom-right corners of the bounding box
            top_left_x = int(center_x - width / 2)
            top_left_y = int(center_y - height / 2)
            bottom_right_x = int(center_x + width / 2)
            bottom_right_y = int(center_y + height / 2)

            # Get the color for this class ID in BGR format
            bbox_color = class_colors_bgr.get(class_id, (0, 0, 0))  # Default to black if class ID not found

            # Get the label name for this class ID
            label = class_labels.get(class_id, 'unknown')

            # Draw the bounding box on the image
            cv2.rectangle(image, (top_left_x, top_left_y), (bottom_right_x, bottom_right_y), bbox_color, 2)
            
            # Put the class label on the bounding box
            # cv2.putText(image, label, (top_left_x, top_left_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bbox_color, 2)

        # Save the annotated image with a new file name
        output_image_path = os.path.join(directory_path, base_name + '_annotated.jpg')
        cv2.imwrite(output_image_path, image)

        print(f'Annotated image saved as {output_image_path}')


if __name__ == "__main__":
        # Define the directory containing the images and annotations
    directory_path = 'ALSS'

    # Define the class labels
    class_labels = {0: 'human', 1: 'elephant', 2: 'giraffe', 3: 'unknown'}
    class_colors = {
        0: (255, 58, 56),    # human
        1: (253, 159, 150),  # elephant
        2: (254, 112, 35),   # giraffe
        3: (252, 173, 60)    # unknown

    }

    class_colors_bgr = {k: (v[2], v[1], v[0]) for k, v in class_colors.items()}
    
    annotated_image(directory_path,class_labels,class_colors_bgr)