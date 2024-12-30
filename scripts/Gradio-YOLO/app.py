

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




def det_seg_inferencs(
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
    if not Path(model_name).is_absolute() and checkpoints_PATH:
        model_name = Path(checkpoints_PATH) / model_name

    model = YOLO(model_name,task=task)


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


def cls_inference(source, device, model_name="yolov8s-cls.pt"):
    model = YOLO(model_name)

    results = model(source=source, device=device)
    results = list(results)[0]
    return results


# YOLOv8图片检测函数
def det_seg_pipeline(
    source,
    model_name,
    task,
    device=0,
    imgsz=640,
    conf=0.3,
    IoU=0.7,
    max_det=100,
    obj_size="所有尺寸",
    det_seg_name=[], #默认使用全局参数
):

    s_obj, m_obj, l_obj = 0, 0, 0

    tgt_size_area_list = []  # 目标面积

    tgt_size_conf_list = []  # 置信度统计
    tgt_size_boxes_list = []  # 边界框统计
    tgt_size_name_list = []  # 类别数量统计
    tgt_size_name_inds_list = []  # 1

    # 模型加载
    if det_seg_name == []:
        det_seg_name = None
    results = det_seg_inferencs(
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

    # 检测参数
    xyxy_list = results.boxes.xyxy.cpu().numpy().tolist() #坐标
    conf_list = results.boxes.conf.cpu().numpy().tolist() #置信度
    name_list = results.boxes.cls.cpu().numpy().tolist()  #类别

    # 颜色列表
    color_list = random_color(len(det_seg_name_list), True)

    img = Image.open(source)
    img_cp = img.copy()

    # 图像分割
    if task == "segment":
        # masks_list = results.masks.xyn
        masks_list = results.masks.xy
        img_mask_merge = seg_output(source, masks_list, color_list, name_list)
        img = Image.fromarray(cv2.cvtColor(img_mask_merge, cv2.COLOR_BGRA2RGB))

    # 处理检测
    if xyxy_list != []:
        # ---------------- 加载字体 ----------------
        yaml_index = det_seg_name_yaml.index(".yaml")
        language = det_seg_name_yaml[yaml_index - 2 : yaml_index]

        # 字体
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
            # ------------ 边框坐标 ------------
            x0 = int(xyxy_list[i][0])
            y0 = int(xyxy_list[i][1])
            x1 = int(xyxy_list[i][2])
            y1 = int(xyxy_list[i][3])

            # ---------- 加入目标尺寸 ----------
            w = x1 - x0
            h = y1 - y0
            area = w * h  # 目标尺寸

            if obj_size == obj_style[0] and area > 0 and area <= 32**2:
                
                tgt_size_name_inds_list.append(int(name_list[i]))
                tgt_size_name_list.append(det_seg_name_list[int(name_list[i])] )
                tgt_size_boxes_list.append((x0, y0, x1, y1))
                tgt_size_conf_list.append(float(conf_list[i]))
                tgt_size_area_list.append(area)
                
            elif (
                obj_size == obj_style[1] and area > 32**2 and area <= 96**2
            ):
                
                tgt_size_name_inds_list.append(int(name_list[i]))
                tgt_size_name_list.append(det_seg_name_list[int(name_list[i])] )
                tgt_size_boxes_list.append((x0, y0, x1, y1))
                tgt_size_conf_list.append(float(conf_list[i]))
                tgt_size_area_list.append(area)
                
            elif obj_size == obj_style[2] and area > 96**2:
                
                tgt_size_name_inds_list.append(int(name_list[i]))
                tgt_size_name_list.append(det_seg_name_list[int(name_list[i])] )
                tgt_size_boxes_list.append((x0, y0, x1, y1))
                tgt_size_conf_list.append(float(conf_list[i]))
                tgt_size_area_list.append(area)
                
            elif obj_size == "所有尺寸":
                
                tgt_size_name_inds_list.append(int(name_list[i]))
                tgt_size_name_list.append(det_seg_name_list[int(name_list[i])] )
                tgt_size_boxes_list.append((x0, y0, x1, y1))
                tgt_size_conf_list.append(float(conf_list[i]))
                tgt_size_area_list.append(area)

        det_img = pil_draw(
            img,
            tgt_size_conf_list,
            tgt_size_boxes_list,
            tgt_size_name_list,
            tgt_size_name_inds_list,
            textFont,
            color_list,
        )

        # -------------- 目标尺寸计算 --------------
        for i in range(len(tgt_size_area_list)):
            if 0 < tgt_size_area_list[i] <= 32**2:
                s_obj = s_obj + 1
            elif 32**2 < tgt_size_area_list[i] <= 96**2:
                m_obj = m_obj + 1
            elif tgt_size_area_list[i] > 96**2:
                l_obj = l_obj + 1

        sml_obj_total = s_obj + m_obj + l_obj
        objSizeRatio_dict = {}
        objSizeRatio_dict = {
            obj_style[i]: [s_obj, m_obj, l_obj][i] / sml_obj_total for i in range(3)
        }

        # ------------ 类别统计 ------------
        clsRatio_dict = {}
        clsDet_dict = Counter(tgt_size_name_list)
        clsDet_dict_sum = sum(clsDet_dict.values())
        for k, v in clsDet_dict.items():
            clsRatio_dict[k] = v / clsDet_dict_sum

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


def cls_pipeline(source, device, model_name):
    # 模型加载
    results = cls_inference(
        source, device, model_name=f"{model_name}"
    )

    det_img = Image.open(source)
    clas_ratio_list = results.probs.top5conf.tolist()
    clas_index_list = results.probs.top5

    clas_name_list = []
    for i in clas_index_list:
        # clas_name_list.append(results.names[i])
        clas_name_list.append(classify_name_list[i])

    clsRatio_dict = {}
    index_cls = 0
    clsDet_dict = Counter(clas_name_list)
    for k, v in clsDet_dict.items():
        clsRatio_dict[k] = clas_ratio_list[index_cls]
        index_cls += 1

    return det_img, clsRatio_dict


def main(conf_yaml = "default.yaml"):

    global det_seg_name_list, classify_name_list, det_seg_name_yaml, classify_name_yaml #实际名字以及名字对应的yaml文件
    
    conf_yaml = "default.yaml"
    if not Path(conf_yaml).is_absolute() and ROOT_PATH: #相对路径
        conf_yaml = Path(ROOT_PATH) / conf_yaml

    with open(conf_yaml, 'r', encoding='utf-8') as f:
        conf_dict = yaml.safe_load(f)
        
        
    gr.close_all()
    # 模型
    support_det_seg_model_yaml = conf_dict.get("support_det_seg_model_yaml")
    support_classify_model_yaml = conf_dict.get("support_classify_model_yaml")
    support_det_seg_model_list = yaml_csv(support_det_seg_model_yaml, "support_models")  
    support_classify_model_list = yaml_csv(support_classify_model_yaml, "support_models")  
    
    # 类别
    det_seg_name_yaml = conf_dict.get("det_seg_name_yaml") #类别名称
    classify_name_yaml = conf_dict.get("classify_name_yaml")
    det_seg_name_list = yaml_csv(det_seg_name_yaml, "name")  # 类别名称
    classify_name_list = yaml_csv(classify_name_yaml, "name")  # 类别名称

    
    
    nms_conf = conf_dict.get("nms_conf")
    IoU = conf_dict.get("IoU")
    imgsz = conf_dict.get("imgsz")
    max_det = conf_dict.get("max_det")
    slider_step = conf_dict.get("slider_step")
    
    check_fonts(f"{ROOT_PATH}/fonts")  # 检查字体文件
    

    

    custom_theme = gr.themes.Soft(primary_hue="blue").set(
        button_secondary_background_fill="*neutral_100",
        button_secondary_background_fill_hover="*neutral_200",
    )

    # ------------ Gradio Blocks ------------
    with gr.Blocks(theme=custom_theme, css=custom_css) as gyd:
        with gr.Row():
            gr.Markdown(TITLE)
        with gr.Row():
            with gr.Tabs():
                with gr.TabItem("目标检测与图像分割"):
                    with gr.Row():
                        with gr.Group(elem_id="show_box"): #理解为初始化
                            with gr.Column(scale=1):
                                with gr.Row():
                                    inputs_sources = gr.Image(
                                        image_mode="RGB", type="filepath", label="输入图片"
                                    )
                                with gr.Row():
                                    inputs_model_name = gr.Dropdown( #下拉选项
                                        choices=support_det_seg_model_list,
                                        value=support_det_seg_model_list[0], #默认模型名
                                        type="value",
                                        label="推理模型",
                                    )
                                with gr.Accordion(open=True):
                                    with gr.Row():
                                        input_task = gr.Radio(
                                            choices=["detect", "segment"],
                                            value="seg",
                                            label="任务",
                                        )
                                    with gr.Row():
                                        input_device = gr.Radio(
                                            choices=["cpu", 0, 1, 2, 3],
                                            value=0,
                                            label="推理设备",
                                        )
                                with gr.Accordion("超参数设置", open=True):
                                    with gr.Row():
                                        input_imgsz = gr.Slider(
                                            320,
                                            1600,
                                            step=1,
                                            value=imgsz,
                                            label="推理图像尺寸",
                                        )
                                        input_max_det = gr.Slider(
                                            1,
                                            1000,
                                            step=1,
                                            value=max_det,
                                            label="最大检测数",
                                        )
                                    with gr.Row():
                                        input_nms_conf = gr.Slider(
                                            0,
                                            1,
                                            step=slider_step,
                                            value=nms_conf,
                                            label="置信度阈值",
                                        )
                                        input_IoU = gr.Slider(
                                            0,
                                            1,
                                            step=slider_step,
                                            value=IoU,
                                            label="IoU 阈值",
                                        )
                                    with gr.Row():
                                        input_obj_size = gr.Radio(
                                            choices=["所有尺寸", "小目标", "中目标", "大目标"],
                                            value="所有尺寸",
                                            label="目标尺寸",
                                        )
                                with gr.Row():
                                    input_det_seg_name = gr.Dropdown(
                                        choices=det_seg_name_list,
                                        value=[], #list
                                        multiselect=True,
                                        allow_custom_value=True,
                                        type="index",
                                        label="类别选择",
                                    )
                                with gr.Row():
                                    gr.ClearButton(inputs_sources, value="清除")
                                    det_seg_button = gr.Button( #绑定键
                                        value="检测", variant="primary"
                                    )

                        with gr.Group(elem_id="show_box"):
                            with gr.Column(scale=1):
                                with gr.Row():
                                    outputs_img_slider = gr.Image(type="pil", label="输出图像")
                                with gr.Row():
                                    outputs_img_paths = gr.Files(label="图片下载")
                                with gr.Row():
                                    outputs_objSize = gr.Label(label="尺寸占比")
                                with gr.Row():
                                    outputs_clsRatio_dict = gr.Label(label="类别占比")

                    # 示例
                    with gr.Group(elem_id="show_box"): #实例也是绑定同一个元素id
                        with gr.Row():
                            gr.Examples(
                                examples=EXAMPLES_DET_SEG,
                                fn=det_seg_pipeline, #回调函数
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

                with gr.TabItem("图像分类"):
                    with gr.Row():
                        with gr.Group(elem_id="show_box"):
                            with gr.Column(scale=1):
                                with gr.Row():
                                    inputs_img_cls = gr.Image(
                                        image_mode="RGB", type="filepath", label="原始图片"
                                    )
                                with gr.Row():
                                    device_opt_cls = gr.Radio(
                                        choices=["cpu", "0", "1", "2", "3"],
                                        value="cpu",
                                        label="设备",
                                    )
                                with gr.Row():
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
                                with gr.Row():
                                    gr.ClearButton(inputs_sources, value="清除")
                                    det_seg_button_cls = gr.Button(
                                        value="检测", variant="primary"
                                    )

                        with gr.Group(elem_id="show_box"):
                            with gr.Column(scale=1):
                                with gr.Row():
                                    outputs_img_cls = gr.Image(type="pil", label="检测图片")
                                with gr.Row():
                                    outputs_ratio_cls = gr.Label(label="图像分类结果")

                    with gr.Group(elem_id="show_box"):
                        with gr.Row():
                            gr.Examples(
                                examples=EXAMPLES_CLAS,
                                fn=cls_pipeline,
                                inputs=[
                                    inputs_img_cls,
                                    device_opt_cls,
                                    inputs_model_cls,
                                ],
                                # outputs=[outputs_img_cls, outputs_ratio_cls],
                                cache_examples=False,
                            )
                
        det_seg_button.click(
            fn=det_seg_pipeline,
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
            outputs=[
                outputs_img_slider,
                outputs_img_paths,
                outputs_objSize,
                outputs_clsRatio_dict,
            ],
        )

        det_seg_button_cls.click(
            fn=cls_pipeline,
            inputs=[inputs_img_cls, device_opt_cls, inputs_model_cls],
            outputs=[outputs_img_cls, outputs_ratio_cls],
        )

    return gyd


if __name__ == "__main__":
    

    gyd = main()


    gyd.queue().launch(
        inbrowser=True,  # 自动打开默认浏览器
        share=False,  # 项目共享，其他设备可以访问
        show_error=True,  # 在浏览器控制台中显示错误信息
        quiet=True,  # 禁止大多数打印语句
    )
