from PIL import Image

def resize_image(input_path, output_path, size=(640, 640)):
    with Image.open(input_path) as img:
        resized_img = img.resize(size, Image.LANCZOS)
        resized_img.save(output_path)

if __name__ == "__main__":
    input_image_path = 'ultralytics/assets/zidane.jpg'
    output_image_path = 'ultralytics/assets/zidane_640.jpg'
    
    resize_image(input_image_path, output_image_path)
    print(f"Image resized and saved to {output_image_path}")
