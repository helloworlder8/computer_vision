import cv2
import os
import glob
import numpy as np

# Function to generate random colors
def random_color():
    return tuple(np.random.randint(0, 256, size=3).tolist())

# Class for predefined color palettes
class ColorPalette:
    def __init__(self, palette=None):
        # Default color palette
        self.pose_palette = np.array(
            palette if palette is not None else [
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

    def get_color(self, index):
        if 0 <= index < len(self.pose_palette):
            return tuple(self.pose_palette[index].tolist())
        else:
            raise IndexError("Index out of bounds. Please provide a valid index.")

# Function to draw custom markers at the center points
def draw_custom_marker(image, center, color, marker_type, size, edgecolor, linewidth):
    x, y = center
    if marker_type == '*':
        # Draw a star shape
        cv2.drawMarker(image, (x, y), color, markerType=cv2.MARKER_STAR, markerSize=size, thickness=linewidth, line_type=cv2.LINE_AA)
    elif marker_type == 'o':
        # Draw a circle with edge color
        cv2.circle(image, (x, y), size // 2, color, thickness=-1)  # Filled circle
        cv2.circle(image, (x, y), size // 2, edgecolor, thickness=linewidth)  # Edge circle
    else:
        # Default to a cross if the marker type is not recognized
        cv2.drawMarker(image, (x, y), color, markerType=cv2.MARKER_CROSS, markerSize=size, thickness=linewidth, line_type=cv2.LINE_AA)

# Main function to draw bounding boxes and optional center points with customization
def annotated_image_detection(
    image_folder, label_folder, output_folder, file_extension="JPG", palette=None, step_size=5, random_color_mode=False,
    draw_center_points=False, marker_type='*', marker_size=10, edgecolor=(255, 255, 255), linewidth=1
):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    image_files = glob.glob(os.path.join(image_folder, f'*.{file_extension}'))
    color_palette = ColorPalette(palette)

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
        num = 8

        for line in lines:
            values = line.split()
            class_id = int(values[0])  # Extract the class ID
            x_center = float(values[1]) * image_width
            y_center = float(values[2]) * image_height
            box_width = float(values[3]) * image_width * 1.2
            box_height = float(values[4]) * image_height * 1.2

            # Calculate the top-left and bottom-right corners of the bounding box
            x1 = max(0, int(x_center - box_width / 2))
            y1 = max(0, int(y_center - box_height / 2))
            x2 = min(int(x_center + box_width / 2),image_width)
            y2 = min(int(y_center + box_height / 2),image_height)

            # Choose the color for the bounding box
            if random_color_mode:
                box_color = random_color()
            else:
                box_color = color_palette.get_color(18)
            num += step_size
            if num>=19:
                num=0
            if class_id == 68 or class_id == 109 or class_id == 72:
                # Draw the bounding box
                # cv2.rectangle(image, (x1, y1), (x2, y2), color=box_color, thickness=4)
                cv2.rectangle(image, (x1, y1), (x2, y2), color=box_color, thickness=2)
                # Draw the center point with custom style if the option is enabled
                # if draw_center_points:
                #     draw_custom_marker(image, (int(x_center), int(y_center)), box_color, marker_type, marker_size, edgecolor, linewidth)

        output_image_path = os.path.join(output_folder, base_name + '_annotatedv2.jpg')
        cv2.imwrite(output_image_path, image)
        print(f'Annotated image saved as {output_image_path}')

if __name__ == "__main__":
    # Parameters can be modified as needed
    image_folder = 'select_image/2/det_labels/MMB+20%/'
    label_folder = 'select_image/2/det_labels/'
    output_folder = 'select_image/2/det_labels/MMB+20%/'
    file_extension = "jpg"  # Image file extension
    palette = None  # Use None for the default palette
    step_size = 5  # How many indices to jump in the palette
    random_color_mode = False  # Set to True if you want to use random colors
    draw_center_points = True  # Set to True to draw center points

    # Custom marker options
    marker_type = '*'  # Supported types: '*', 'o', etc.
    marker_size = 2  # Marker size
    edgecolor = (255, 255, 255)  # White edge color
    linewidth = 3  # Line thickness

    annotated_image_detection(
        image_folder=image_folder,
        label_folder=label_folder,
        output_folder=output_folder,
        file_extension=file_extension,
        palette=palette,
        step_size=step_size,
        random_color_mode=random_color_mode,
        draw_center_points=draw_center_points,
        marker_type=marker_type,
        marker_size=marker_size,
        edgecolor=edgecolor,
        linewidth=linewidth
    )
