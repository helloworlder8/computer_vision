
import os
import sys
from pathlib import Path
from matplotlib import font_manager
import wget
from rich.console import Console
import csv
import yaml
import numpy as np
import argparse
import csv
import random
import sys
import cv2
from PIL import ImageDraw


ROOT_PATH = sys.path[0]  # 项目根目录

# 中文、英文、俄语、西班牙语、阿拉伯语、韩语
fonts_list = ["SimSun.ttf", "TimesNewRoman.ttf", "malgun.ttf"]  # 字体列表
fonts_suffix = ["ttc", "ttf", "otf"]  # 字体后缀

data_url_dict = {
    "SimSun.ttf": "https://gitee.com/CV_Lab/gradio_yolov5_det/attach_files/1053539/download/SimSun.ttf",
    "TimesNewRoman.ttf": "https://gitee.com/CV_Lab/gradio_yolov5_det/attach_files/1053537/download/TimesNewRoman.ttf",
    "malgun.ttf": "https://gitee.com/CV_Lab/gradio_yolov5_det/attach_files/1053538/download/malgun.ttf",}

console = Console()



ROOT_PATH = sys.path[0]  #表示当前执行文件的母文件夹

checkpoints_PATH = os.path.join(ROOT_PATH, "checkpoints/")


# 文件后缀
suffix_list = [".csv", ".yaml"]

# 字体大小
FONTSIZE = 25

# 目标尺寸
obj_style = ["小目标", "中目标", "大目标"]

TITLE = """
<p align='center'>

<a href='https://github.com/helloworlder8/computer_vision'>
个人GitHub主页</a>
<p align='center' style="font-size:30px;font-weight:700">锐明像素演示系统</p>

</p>
"""




EXAMPLES_DET_SEG = [
    [Path(ROOT_PATH) / "img_examples" / "0000000353_0000000000_0000000652.jpg", "ALSS-YOLO-seg.pt","segment"],
]

EXAMPLES_CLAS = [
    [Path(ROOT_PATH) / "img_examples" / "ILSVRC2012_val_00000008.JPEG", 0, "yolov8n-cls"],
    [Path(ROOT_PATH) / "img_examples" / "ILSVRC2012_val_00000018.JPEG", 0, "yolov8n-cls"],
    [Path(ROOT_PATH) / "img_examples" / "ILSVRC2012_val_00000023.JPEG", 0, "yolov8n-cls"],
    [Path(ROOT_PATH) / "img_examples" / "ILSVRC2012_val_00000067.JPEG", 0, "yolov8n-cls"],
    [Path(ROOT_PATH) / "img_examples" / "ILSVRC2012_val_00000077.JPEG", 0, "yolov8n-cls"],
]

GYD_CSS = """#disp_image {
        text-align: center; /* Horizontally center the content */
    }"""


custom_css = "./gyd_style.css"



SimSun_path = f"{ROOT_PATH}/fonts/SimSun.ttf"  
TimesNesRoman_path = f"{ROOT_PATH}/fonts/TimesNewRoman.ttf"  
SimSun = font_manager.FontProperties(fname=SimSun_path, size=12)
TimesNesRoman = font_manager.FontProperties(fname=TimesNesRoman_path, size=12)


# yaml文件解析
def yaml_parse(file_path):
    return yaml.safe_load(open(file_path, encoding="utf-8").read())

# yaml csv 文件解析 ROOT_PATH
def yaml_csv(file_path, key):
    # 判断是否是相对路径，如果是则拼接ROOT_PATH
    if not Path(file_path).is_absolute() and ROOT_PATH:
        file_path = Path(ROOT_PATH) / file_path

    file_suffix = Path(file_path).suffix
    if file_suffix == suffix_list[0]:
        # 处理 CSV 文件
        file_list = [i[0] for i in list(csv.reader(open(file_path)))]  # csv版
    elif file_suffix == suffix_list[1]:
        # 处理 YAML 文件
        file_list = yaml_parse(file_path).get(key)  # yaml版
    else:
        print(f"{file_path} 格式不正确！程序退出！")
        sys.exit()

    return file_list



# 创建字体库
def add_fronts(font_diff):

    global font_name

    for k, v in data_url_dict.items():
        if k in font_diff:
            font_name = v.split("/")[-1]  # 字体名称
            Path(f"{ROOT_PATH}/fonts").mkdir(parents=True, exist_ok=True)  # 创建目录

            file_path = f"{ROOT_PATH}/fonts/{font_name}"  # 字体路径

            try:
                # 下载字体文件
                wget.download(v, file_path)
            except Exception as e:
                print("路径错误！程序结束！")
                print(e)
                sys.exit()
            else:
                print()
                console.print(f"{font_name} [bold green]字体文件下载完成！[/bold green] 已保存至：{file_path}")


# 判断字体文件
def check_fonts(fonts_dir):
    if os.path.isdir(fonts_dir):
        # 如果字体库存在
        f_list = os.listdir(fonts_dir)  # 本地字体库

        font_diff = list(set(fonts_list).difference(set(f_list)))

        if font_diff != []:
            # 字体不存在
            console.print("[bold red]字体不存在，正在加载。。。[/bold red]")
            add_fronts(font_diff)  # 创建字体库
        else:
            console.print(f"{fonts_list}[bold green]字体已存在！[/bold green]")
    else:
        # 字体库不存在，创建字体库
        console.print("[bold red]字体库不存在，正在创建。。。[/bold red]")
        add_fronts(fonts_list)  # 创建字体库



def check_online():
    import socket
    try:
        socket.create_connection(("1.1.1.1", 443), 5)  # check host accessibility
        return True
    except OSError:
        return False








# 标签和边界框颜色设置
def color_set(cls_num):
    color_list = []
    for i in range(cls_num):
        color = tuple(np.random.choice(range(256), size=3))
        color_list.append(color)

    return color_list


# 随机生成浅色系或者深色系
def random_color(cls_num, is_light=True):
    color_list = []
    for i in range(cls_num):
        color = (
            random.randint(0, 127) + int(is_light) * 128,
            random.randint(0, 127) + int(is_light) * 128,
            random.randint(0, 127) + int(is_light) * 128,
        )
        color_list.append(color)

    return color_list


# 检测绘制
def pil_draw(img, tgt_size_conf_list, tgt_size_boxes_list, tgt_size_name_list, tgt_size_name_inds_list, textFont, color_list):
    img_pil = ImageDraw.Draw(img)
    id = 0

    for conf_i, (xmin_i, ymin_i, xmax_i, ymax_i), name_i, name_inds_i in zip(
        tgt_size_conf_list, tgt_size_boxes_list, tgt_size_name_list, tgt_size_name_inds_list
    ):
        img_pil.rectangle(
            [xmin_i, ymin_i, xmax_i, ymax_i], fill=None, outline=color_list[name_inds_i], width=2
        )  # 边界框
        countdown_msg = f"{id}-{name_i} {conf_i:.2f}"
        # text_w, text_h = textFont.getsize(countdown_msg)  # 标签尺寸 pillow 9.5.0
        # left, top, left + width, top + height
        # 标签尺寸 pillow 10.0.0
        text_xmin, text_ymin, text_xmax, text_ymax = textFont.getbbox(countdown_msg)
        # 标签背景
        img_pil.rectangle(
            # (xmin_i, ymin_i, xmin_i + text_w, ymin_i + text_h), # pillow 9.5.0
            (
                xmin_i,
                ymin_i,
                xmin_i + text_xmax - text_xmin,
                ymin_i + text_ymax - text_ymin,
            ),  # pillow 10.0.0
            fill=color_list[name_inds_i],
            outline=color_list[name_inds_i],
        )

        # 标签
        img_pil.multiline_text(
            (xmin_i, ymin_i),
            countdown_msg,
            fill=(0, 0, 0),
            font=textFont,
            align="center",
        )

        id += 1

    return img


# 绘制多边形
def polygon_drawing(img_mask, canvas, color_seg):
    # ------- RGB转BGR -------
    color_seg = list(color_seg)
    color_seg[0], color_seg[2] = color_seg[2], color_seg[0]
    color_seg = tuple(color_seg)
    # 定义多边形的顶点
    pts = np.array(img_mask, dtype=np.int32)

    # 多边形绘制
    cv2.drawContours(canvas, [pts], -1, color_seg, thickness=-1)


# 输出分割结果
def seg_output(source, seg_mask_list, color_list, name_list):
    img = cv2.imread(source)
    img_c = img.copy()

    # w, h = img.shape[1], img.shape[0]

    # 获取分割坐标
    for seg_mask, cls_index in zip(seg_mask_list, name_list):
        img_mask = []
        for i in range(len(seg_mask)):
            # img_mask.append([seg_mask[i][0] * w, seg_mask[i][1] * h])
            img_mask.append([seg_mask[i][0], seg_mask[i][1]])

        polygon_drawing(img_mask, img_c, color_list[int(cls_index)])  # 绘制分割图形

    img_mask_merge = cv2.addWeighted(img, 0.3, img_c, 0.7, 0)  # 合并图像

    return img_mask_merge
