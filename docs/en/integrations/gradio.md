---
comments: true
description: Discover an interactive way to perform object detection with Ultralytics YOLOv8 using Gradio. Upload images and adjust settings for real-time results.
keywords: Ultralytics, YOLOv8, Gradio, object detection, interactive, real-time, image processing, AI
---

# Interactive Object Detection: Gradio & Ultralytics YOLOv8 🚀

## Introduction to Interactive Object Detection

This Gradio interface provides an easy and interactive way to perform object detection using the [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics/) model. Users can upload images and adjust parameters like confidence threshold and intersection-over-union (IoU) threshold to get real-time detection results.

<p align="center">
  <br>
  <iframe loading="lazy" width="720" height="405" src="https://www.youtube.com/embed/pWYiene9lYw"
    title="YouTube video player" frameborder="0"
    allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share"
    allowfullscreen>
  </iframe>
  <br>
  <strong>Watch:</strong> Gradio Integration with Ultralytics YOLOv8
</p>

## Why Use Gradio for Object Detection?

- **User-Friendly Interface:** Gradio offers a straightforward platform for users to upload images and visualize detection results without any coding requirement.
- **Real-Time Adjustments:** Parameters such as confidence and IoU thresholds can be adjusted on the fly, allowing for immediate feedback and optimization of detection results.
- **Broad Accessibility:** The Gradio web interface can be accessed by anyone, making it an excellent tool for demonstrations, educational purposes, and quick experiments.

<p align="center">
   <img width="800" alt="Gradio example screenshot" src="https://github.com/RizwanMunawar/ultralytics/assets/26833433/52ee3cd2-ac59-4c27-9084-0fd05c6c33be">
</p>

## How to Install the Gradio

```bash
pip install gradio
```

## How to Use the Interface

1. **Upload Image:** Click on 'Upload Image' to choose an image file for object detection.
2. **Adjust Parameters:**
    - **Confidence Threshold:** Slider to set the minimum confidence level for detecting objects.
    - **IoU Threshold:** Slider to set the IoU threshold for distinguishing different objects.
3. **View Results:** The processed image with detected objects and their labels will be displayed.

## Example Use Cases

- **Sample Image 1:** Bus detection with default thresholds.
- **Sample Image 2:** Detection on a sports image with default thresholds.

## Usage Example

This section provides the Python code used to create the Gradio interface with the Ultralytics YOLOv8 model. Supports classification tasks, detection tasks, segmentation tasks, and key point tasks.

```python
import gradio as gr
import PIL.Image as Image

from ultralytics import ASSETS, YOLO

model = YOLO("yolov8n.pt")


def predict_image(img, conf_threshold, iou_threshold):
    """Predicts objects in an image using a YOLOv8 model with adjustable confidence and IOU thresholds."""
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
        imgsz=640,
    )

    for r in results:
        im_array = r.plot()
        img = Image.fromarray(im_array[..., ::-1])

    return img


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Ultralytics Gradio",
    description="Upload images for inference. The Ultralytics YOLOv8n model is used by default.",
    examples=[
        [ASSETS / "bus.jpg", 0.25, 0.45],
        [ASSETS / "zidane.jpg", 0.25, 0.45],
    ],
)

if __name__ == "__main__":
    iface.launch()
```

## Parameters Explanation

| Parameter Name   | Type    | Description                                              |
| ---------------- | ------- | -------------------------------------------------------- |
| `img`            | `Image` | The image on which object detection will be performed.   |
| `conf_threshold` | `float` | Confidence threshold for detecting objects.              |
| `iou_threshold`  | `float` | Intersection-over-union threshold for object separation. |

### Gradio Interface Components

| Component    | Description                              |
| ------------ | ---------------------------------------- |
| Image Input  | To upload the image for detection.       |
| Sliders      | To adjust confidence and IoU thresholds. |
| Image Output | To display the detection results.        |

## FAQ

### How do I use Gradio with Ultralytics YOLOv8 for object detection?

To use Gradio with Ultralytics YOLOv8 for object detection, you can follow these steps:

1. **Install Gradio:** Use the command `pip install gradio`.
2. **Create Interface:** Write a Python script to initialize the Gradio interface. You can refer to the provided code example in the [documentation](#usage-example) for details.
3. **Upload and Adjust:** Upload your image and adjust the confidence and IoU thresholds on the Gradio interface to get real-time object detection results.

Here's a minimal code snippet for reference:

```python
import gradio as gr

from ultralytics import YOLO

model = YOLO("yolov8n.pt")


def predict_image(img, conf_threshold, iou_threshold):
    results = model.predict(
        source=img,
        conf=conf_threshold,
        iou=iou_threshold,
        show_labels=True,
        show_conf=True,
    )
    return results[0].plot() if results else None


iface = gr.Interface(
    fn=predict_image,
    inputs=[
        gr.Image(type="pil", label="Upload Image"),
        gr.Slider(minimum=0, maximum=1, value=0.25, label="Confidence threshold"),
        gr.Slider(minimum=0, maximum=1, value=0.45, label="IoU threshold"),
    ],
    outputs=gr.Image(type="pil", label="Result"),
    title="Ultralytics Gradio YOLOv8",
    description="Upload images for YOLOv8 object detection.",
)
iface.launch()
```

### What are the benefits of using Gradio for Ultralytics YOLOv8 object detection?

Using Gradio for Ultralytics YOLOv8 object detection offers several benefits:

- **User-Friendly Interface:** Gradio provides an intuitive interface for users to upload images and visualize detection results without any coding effort.
- **Real-Time Adjustments:** You can dynamically adjust detection parameters such as confidence and IoU thresholds and see the effects immediately.
- **Accessibility:** The web interface is accessible to anyone, making it useful for quick experiments, educational purposes, and demonstrations.

For more details, you can read this [blog post](https://www.ultralytics.com/blog/ai-and-radiology-a-new-era-of-precision-and-efficiency).

### Can I use Gradio and Ultralytics YOLOv8 together for educational purposes?

Yes, Gradio and Ultralytics YOLOv8 can be utilized together for educational purposes effectively. Gradio's intuitive web interface makes it easy for students and educators to interact with state-of-the-art deep learning models like Ultralytics YOLOv8 without needing advanced programming skills. This setup is ideal for demonstrating key concepts in object detection and computer vision, as Gradio provides immediate visual feedback which helps in understanding the impact of different parameters on the detection performance.

### How do I adjust the confidence and IoU thresholds in the Gradio interface for YOLOv8?

In the Gradio interface for YOLOv8, you can adjust the confidence and IoU thresholds using the sliders provided. These thresholds help control the prediction accuracy and object separation:

- **Confidence Threshold:** Determines the minimum confidence level for detecting objects. Slide to increase or decrease the confidence required.
- **IoU Threshold:** Sets the intersection-over-union threshold for distinguishing between overlapping objects. Adjust this value to refine object separation.

For more information on these parameters, visit the [parameters explanation section](#parameters-explanation).

### What are some practical applications of using Ultralytics YOLOv8 with Gradio?

Practical applications of combining Ultralytics YOLOv8 with Gradio include:

- **Real-Time Object Detection Demonstrations:** Ideal for showcasing how object detection works in real-time.
- **Educational Tools:** Useful in academic settings to teach object detection and computer vision concepts.
- **Prototype Development:** Efficient for developing and testing prototype object detection applications quickly.
- **Community and Collaborations:** Making it easy to share models with the community for feedback and collaboration.

For examples of similar use cases, check out the [Ultralytics blog](https://www.ultralytics.com/blog/monitoring-animal-behavior-using-ultralytics-yolov8).

Providing this information within the documentation will help in enhancing the usability and accessibility of Ultralytics YOLOv8, making it more approachable for users at all levels of expertise.
