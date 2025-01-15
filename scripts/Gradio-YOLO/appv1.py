

from collections import Counter
from pathlib import Path

import cv2
import gradio as gr
from gradio_imageslider import ImageSlider
import tempfile
import uuid


from ultralytics import YOLO
import yaml
from PIL import Image, ImageDraw, ImageFont

from util import *


class GradioApp:
    def __init__(self, conf_yaml="default.yaml"):
        self.conf_yaml = conf_yaml
        self.load_config()
        self.setup_globals()
        self.custom_theme = self.create_theme()
        self.build_interface()

    def load_config(self):
        if not Path(self.conf_yaml).is_absolute() and 'ROOT_PATH' in globals():
            self.conf_yaml = Path(ROOT_PATH) / self.conf_yaml

        with open(self.conf_yaml, 'r', encoding='utf-8') as f:
            self.conf_dict = yaml.safe_load(f)

    def setup_globals(self):
        # Close any existing Gradio sessions
        gr.close_all()

        # Model configurations
        self.support_det_seg_model_yaml = self.conf_dict.get("support_det_seg_model_yaml")
        self.support_classify_model_yaml = self.conf_dict.get("support_classify_model_yaml")
        self.support_det_seg_model_list = yaml_csv(self.support_det_seg_model_yaml, "support_models")
        self.support_classify_model_list = yaml_csv(self.support_classify_model_yaml, "support_models")

        # Category names
        self.det_seg_name_yaml = self.conf_dict.get("det_seg_name_yaml")
        self.classify_name_yaml = self.conf_dict.get("classify_name_yaml")
        self.det_seg_name_list = yaml_csv(self.det_seg_name_yaml, "name")
        self.classify_name_list = yaml_csv(self.classify_name_yaml, "name")

        # Hyperparameters
        self.nms_conf = self.conf_dict.get("nms_conf")
        self.IoU = self.conf_dict.get("IoU")
        self.imgsz = self.conf_dict.get("imgsz")
        self.max_det = self.conf_dict.get("max_det")
        self.slider_step = self.conf_dict.get("slider_step")

        # Check fonts
        check_fonts(f"{ROOT_PATH}/fonts")

    def create_theme(self):
        return gr.themes.Soft(primary_hue="blue").set(
            button_secondary_background_fill="*neutral_100",
            button_secondary_background_fill_hover="*neutral_200",
        )

    def build_interface(self):
        with gr.Blocks(theme=self.custom_theme, css=custom_css) as self.app:
            self.add_title()
            self.add_tabs()
            self.add_event_handlers()
        # Assign to self.app for external access if needed

    def add_title(self):
        with gr.Row():
            gr.Markdown(TITLE)

    def add_tabs(self):
        with gr.Row():
            with gr.Tabs():
                self.add_det_seg_tab()
                self.add_classify_tab()

    def add_det_seg_tab(self):
        with gr.TabItem("目标检测与图像分割"):
            with gr.Row():
                # Input Components
                with gr.Group(elem_id="show_box"):
                    inputs_sources = gr.Image(
                        image_mode="RGB", type="filepath", label="输入图片"
                    )
                    inputs_model_name = gr.Dropdown(
                        choices=self.support_det_seg_model_list,
                        value=self.support_det_seg_model_list[0],
                        type="value",
                        label="推理模型",
                    )
                    input_task = gr.Radio(
                        choices=["detect", "segment"],
                        value="seg",
                        label="任务",
                    )
                    input_device = gr.Radio(
                        choices=["cpu", 0, 1, 2, 3],
                        value=0,
                        label="推理设备",
                    )
                    input_imgsz = gr.Slider(
                        320,
                        1600,
                        step=1,
                        value=self.imgsz,
                        label="推理图像尺寸",
                    )
                    input_nms_conf = gr.Slider(
                        0,
                        1,
                        step=self.slider_step,
                        value=self.nms_conf,
                        label="置信度阈值",
                    )
                    input_IoU = gr.Slider(
                        0,
                        1,
                        step=self.slider_step,
                        value=self.IoU,
                        label="IoU 阈值",
                    )
                    input_max_det = gr.Slider(
                        1,
                        1000,
                        step=1,
                        value=self.max_det,
                        label="最大检测数",
                    )
                    input_obj_size = gr.Radio(
                        choices=["所有尺寸", "小目标", "中目标", "大目标"],
                        value="所有尺寸",
                        label="目标尺寸",
                    )
                    input_det_seg_name = gr.Dropdown(
                        choices=self.det_seg_name_list,
                        value=[],  # list
                        multiselect=True,
                        allow_custom_value=True,
                        type="index",
                        label="类别选择",
                    )
                    det_seg_button = gr.Button(
                        value="检测", variant="primary"
                    )
                    gr.ClearButton(inputs_sources, value="清除")

                # Output Components
                with gr.Group(elem_id="show_box"):
                    outputs_img_slider = gr.Image(type="pil", label="输出图像")
                    outputs_img_paths = gr.Files(label="图片下载")
                    outputs_objSize = gr.Label(label="尺寸占比")
                    outputs_clsRatio_dict = gr.Label(label="类别占比")

            # Examples
            with gr.Group(elem_id="show_box"):
                gr.Examples(
                    examples=EXAMPLES_DET_SEG,
                    fn=self.det_seg_pipeline,
                    inputs=[
                        inputs_sources,
                        inputs_model_name,
                        input_task,
                        input_device,
                        input_imgsz,
                        input_nms_conf,
                        input_IoU,
                        input_max_det,
                        input_obj_size,
                        input_det_seg_name,
                    ],
                    outputs=[outputs_img_slider, outputs_img_paths, outputs_objSize, outputs_clsRatio_dict],
                    cache_examples=False,
                )

            # Store components for event binding
            self.det_seg_components = {
                "inputs": [
                    inputs_sources,
                    inputs_model_name,
                    input_task,
                    input_device,
                    input_imgsz,
                    input_nms_conf,
                    input_IoU,
                    input_max_det,
                    input_obj_size,
                    input_det_seg_name,
                ],
                "outputs": [
                    outputs_img_slider,
                    outputs_img_paths,
                    outputs_objSize,
                    outputs_clsRatio_dict,
                ],
                "button": det_seg_button,
            }

    def add_classify_tab(self):
        with gr.TabItem("图像分类"):
            with gr.Row():
                # Input Components
                with gr.Group(elem_id="show_box"):
                    inputs_img_cls = gr.Image(
                        image_mode="RGB", type="filepath", label="原始图片"
                    )
                    device_opt_cls = gr.Radio(
                        choices=["cpu", "0", "1", "2", "3"],
                        value="cpu",
                        label="设备",
                    )
                    inputs_model_cls = gr.Dropdown(
                        choices=[
                            "yolov8n-cls",
                            "yolov8s-cls",
                            "yolov8l-cls",
                            "yolov8m-cls",
                            "yolov8x-cls",
                        ],
                        value="yolov8s-cls",
                        type="value",
                        label="模型",
                    )
                    det_seg_button_cls = gr.Button(
                        value="检测", variant="primary"
                    )
                    gr.ClearButton(inputs_img_cls, value="清除")

                # Output Components
                with gr.Group(elem_id="show_box"):
                    outputs_img_cls = gr.Image(type="pil", label="检测图片")
                    outputs_ratio_cls = gr.Label(label="图像分类结果")

            # Examples
            with gr.Group(elem_id="show_box"):
                gr.Examples(
                    examples=EXAMPLES_CLAS,
                    fn=self.cls_pipeline,
                    inputs=[
                        inputs_img_cls,
                        device_opt_cls,
                        inputs_model_cls,
                    ],
                    outputs=[outputs_img_cls, outputs_ratio_cls],
                    cache_examples=False,
                )

            # Store components for event binding
            self.classify_components = {
                "inputs": [
                    inputs_img_cls,
                    device_opt_cls,
                    inputs_model_cls,
                ],
                "outputs": [
                    outputs_img_cls,
                    outputs_ratio_cls,
                ],
                "button": det_seg_button_cls,
            }

    def add_event_handlers(self):
        # Bind detection/segmentation button
        det_seg = self.det_seg_components
        det_seg["button"].click(
            fn=self.det_seg_pipeline,
            inputs=det_seg["inputs"],
            outputs=det_seg["outputs"],
        )

        # Bind classification button
        classify = self.classify_components
        classify["button"].click(
            fn=self.cls_pipeline,
            inputs=classify["inputs"],
            outputs=classify["outputs"],
        )

    def det_seg_inferencs(
        self,
        source,
        model_name,
        task,
        device,
        imgsz,
        conf,
        IoU,
        max_det,
        det_seg_name
    ):
        """Perform detection and segmentation inference using YOLO."""
        if not Path(model_name).is_absolute() and checkpoints_PATH:
            model_name = Path(checkpoints_PATH) / model_name

        model = YOLO(model_name, task=task)

        results = model(
            source=source,
            task=task,
            device=device,
            imgsz=imgsz,
            conf=conf,
            IoU=IoU,
            max_det=max_det,
            classes=det_seg_name,
        )
        results = list(results)[0]
        return results

    def cls_inference(self, source, device, model_name="yolov8s-cls.pt"):
        """Perform classification inference using YOLO."""
        if not Path(model_name).is_absolute() and checkpoints_PATH:
            model_name = Path(checkpoints_PATH) / model_name

        model = YOLO(model_name)

        results = model(source=source, device=device)
        results = list(results)[0]
        return results

    def det_seg_pipeline(
        self,
        source,
        model_name,
        task,
        device=0,
        imgsz=640,
        conf=0.3,
        IoU=0.7,
        max_det=100,
        obj_size="所有尺寸",
        det_seg_name=[],  # default to empty list
    ):
        """Pipeline for detection and segmentation tasks."""
        s_obj, m_obj, l_obj = 0, 0, 0

        tgt_size_area_list = []  # Target area
        tgt_size_conf_list = []  # Confidence scores
        tgt_size_boxes_list = []  # Bounding boxes
        tgt_size_name_list = []   # Category names
        tgt_size_name_inds_list = []  # Category indices

        # Model inference
        if not det_seg_name:
            det_seg_name = None
        results = self.det_seg_inferencs(
            source,
            model_name,
            task,
            device,
            imgsz,
            conf,
            IoU,
            max_det,
            det_seg_name
        )

        # Extract detection results
        xyxy_list = results.boxes.xyxy.cpu().numpy().tolist()  # Coordinates
        conf_list = results.boxes.conf.cpu().numpy().tolist()  # Confidence
        name_list = results.boxes.cls.cpu().numpy().tolist()   # Class indices

        # Generate color list
        color_list = random_color(len(self.det_seg_name_list), True)

        img = Image.open(source)
        img_cp = img.copy()

        # Image segmentation
        if task == "segment":
            masks_list = results.masks.xy  # Adjust based on your YOLO implementation
            img_mask_merge = seg_output(source, masks_list, color_list, name_list)
            img = Image.fromarray(cv2.cvtColor(img_mask_merge, cv2.COLOR_BGRA2RGB))

        # Process detections
        if xyxy_list:
            # Determine language based on YAML file name
            yaml_index = self.det_seg_name_yaml.index(".yaml")
            language = self.det_seg_name_yaml[yaml_index - 2 : yaml_index]

            # Load appropriate font
            if language == "zh":
                # 中文
                textFont = ImageFont.truetype(
                    str(f"{ROOT_PATH}/fonts/SimSun.ttf"), size=FONTSIZE
                )
            elif language in ["en", "ru", "es", "ar"]:
                # 英文、俄语、西班牙语、阿拉伯语
                textFont = ImageFont.truetype(
                    str(f"{ROOT_PATH}/fonts/TimesNewRoman.ttf"), size=FONTSIZE
                )
            elif language == "ko":
                # 韩语
                textFont = ImageFont.truetype(
                    str(f"{ROOT_PATH}/fonts/malgun.ttf"), size=FONTSIZE
                )

            for i in range(len(xyxy_list)):
                # Bounding box coordinates
                x0, y0, x1, y1 = map(int, xyxy_list[i])

                # Calculate area
                w = x1 - x0
                h = y1 - y0
                area = w * h

                # Filter based on object size
                if obj_size == obj_style[0] and 0 < area <= 32**2:
                    pass
                elif obj_size == obj_style[1] and 32**2 < area <= 96**2:
                    pass
                elif obj_size == obj_style[2] and area > 96**2:
                    pass
                elif obj_size == "所有尺寸":
                    pass
                else:
                    continue  # Skip objects that do not match the size criteria

                # Append relevant data
                tgt_size_name_inds_list.append(int(name_list[i]))
                tgt_size_name_list.append(self.det_seg_name_list[int(name_list[i])])
                tgt_size_boxes_list.append((x0, y0, x1, y1))
                tgt_size_conf_list.append(float(conf_list[i]))
                tgt_size_area_list.append(area)

            # Draw bounding boxes and labels on the image
            det_img = pil_draw(
                img,
                tgt_size_conf_list,
                tgt_size_boxes_list,
                tgt_size_name_list,
                tgt_size_name_inds_list,
                textFont,
                color_list,
            )

            # Calculate object size ratios
            for area in tgt_size_area_list:
                if 0 < area <= 32**2:
                    s_obj += 1
                elif 32**2 < area <= 96**2:
                    m_obj += 1
                elif area > 96**2:
                    l_obj += 1

            sml_obj_total = s_obj + m_obj + l_obj
            if sml_obj_total > 0:
                objSizeRatio_dict = {
                    obj_style[i]: [s_obj, m_obj, l_obj][i] / sml_obj_total for i in range(3)
                }
            else:
                objSizeRatio_dict = {style: 0 for style in obj_style}

            # Calculate class ratios
            clsRatio_dict = {}
            clsDet_dict = Counter(tgt_size_name_list)
            clsDet_dict_sum = sum(clsDet_dict.values())
            for k, v in clsDet_dict.items():
                clsRatio_dict[k] = v / clsDet_dict_sum if clsDet_dict_sum > 0 else 0

            # Save images and prepare paths for download
            images = (det_img, img_cp)
            images_names = ("det", "raw")
            images_path = tempfile.mkdtemp()
            images_paths = []
            uuid_name = uuid.uuid4()
            for image, image_name in zip(images, images_names):
                image.save(images_path + f"/img_{uuid_name}_{image_name}.jpg")
                images_paths.append(images_path + f"/img_{uuid_name}_{image_name}.jpg")

            gr.Info("图片检测成功！")
            return det_img, images_paths, objSizeRatio_dict, clsRatio_dict

        else:
            raise gr.Error("图片检测失败！")

    def cls_pipeline(self, source, device, model_name):
        """Pipeline for image classification tasks."""
        # Model inference
        results = self.cls_inference(
            source, device, model_name=model_name
        )

        det_img = Image.open(source)
        clas_ratio_list = results.probs.top5conf.tolist()
        clas_index_list = results.probs.top5.tolist()

        clas_name_list = []
        for i in clas_index_list:
            if i < len(self.classify_name_list):
                clas_name_list.append(self.classify_name_list[i])
            else:
                clas_name_list.append("Unknown")

        clsRatio_dict = {}
        clsDet_dict = Counter(clas_name_list)
        clsDet_dict_sum = sum(clsDet_dict.values())
        for k, v in clsDet_dict.items():
            clsRatio_dict[k] = v / clsDet_dict_sum if clsDet_dict_sum > 0 else 0

        return det_img, clsRatio_dict

    def launch(self, **kwargs):
        self.app.queue().launch(
            inbrowser=True,  # 自动打开默认浏览器
            share=False,     # 项目共享，其他设备可以访问
            show_error=True, # 在浏览器控制台中显示错误信息
            quiet=True,      # 禁止大多数打印语句
            **kwargs
        )


app = GradioApp()
app.launch()