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

# Function to draw filled polygon with transparency
def draw_filled_polygon_with_transparency(image, points, label, polygon_color, alpha=0.5):
    overlay = image.copy()
    cv2.fillPoly(overlay, [points], color=polygon_color)
    cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0, image)

# Main annotation function
def annotated_image_segmentation(
    image_folder, label_folder, output_folder, file_extension="JPG", alpha=0.5,
    palette=None, step_size=5, random_color_mode=False
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
        num = 3

        for line in lines:
            values = line.split()
            label = values[0]
            points = np.array([float(v) for v in values[1:]]).reshape(-1, 2)
            points[:, 0] *= image_width
            points[:, 1] *= image_height
            points[:, 0] = np.clip(points[:, 0], 0, image_width)
            points[:, 1] = np.clip(points[:, 1], 0, image_height)
            points = points.astype(np.int32)

            if random_color_mode:
                polygon_color = random_color()
            else:
                polygon_color = color_palette.get_color(num)

            num += step_size
            if num>=19:
                num=0
            # if label == '68' or label == '109' or label == '72':
            # Draw the polygon outline
            cv2.polylines(image, [points], isClosed=True, color=polygon_color, thickness=2)

            if True:
                draw_filled_polygon_with_transparency(image, points, label, polygon_color, alpha=alpha)

                        
            if False:
                label_position = (points[0][0], points[0][1] - 10)  # 标签显示在第一个顶点的上方
                cv2.putText(image, label, label_position, cv2.FONT_HERSHEY_SIMPLEX, 0.6, polygon_color, 2)


        output_image_path = os.path.join(output_folder, base_name + '.jpg')
        cv2.imwrite(output_image_path, image)
        print(f'Annotated image saved as {output_image_path}')

if __name__ == "__main__":
    # Parameters can be modified as needed
    image_folder = 'result/13'
    label_folder = 'result/13'
    output_folder = 'result/13/gt'
    file_extension = "JPG"  # Image file extension
    alpha = 0.5  # Transparency level
    palette = None  # Use None for the default palette
    step_size = 2  # How many indices to jump in the palette
    random_color_mode = False  # Set to True if you want to use random colors

    annotated_image_segmentation(
        image_folder=image_folder,
        label_folder=label_folder,
        output_folder=output_folder,
        file_extension=file_extension,
        alpha=alpha,
        palette=palette,
        step_size=step_size,
        random_color_mode=random_color_mode
    )
