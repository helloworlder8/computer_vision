# Ultralytics YOLO 🚀, AGPL-3.0 license

import subprocess

import pytest
from PIL import Image

from tests import CUDA_DEVICE_COUNT, CUDA_IS_AVAILABLE
from ultralytics.cfg_yaml import TASK2DATA, TASK2MODEL, TASKS
from ultralytics.utils import ASSETS, WEIGHTS_DIR, checks

# Constants
TASK_MODEL_DATA = [(task, WEIGHTS_DIR / TASK2MODEL[task], TASK2DATA[task]) for task in TASKS]
MODELS = [WEIGHTS_DIR / TASK2MODEL[task] for task in TASKS]


def run(cmd):
    """Execute a shell command using subprocess."""
    subprocess.run(cmd.split(), check=True)


def test_special_modes():
    """Test various special command-line modes for YOLO functionality."""
    run("yolo help")
    run("yolo checks")
    run("yolo version")
    run("yolo settings reset")
    run("yolo cfg")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_train(task, model, data):
    """Test YOLO training for different tasks, models, and datasets."""
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 cache=disk")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_val(task, model, data):
    """Test YOLO validation process for specified task, model, and data using a shell command."""
    run(f"yolo val {task} model={model} data={data} imgsz=32 save_txt save_json")


@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
def test_predict(task, model, data):
    """Test YOLO prediction on provided sample assets for specified task and model."""
    run(f"yolo predict model={model} source={ASSETS} imgsz=32 save save_crop save_txt")


@pytest.mark.parametrize("model", MODELS)
def test_export(model):
    """Test exporting a YOLO model to TorchScript format."""
    run(f"yolo export model={model} format=torchscript imgsz=32")


def test_rtdetr(task="detect", model="yolov8n-rtdetr.yaml", data="coco8.yaml"):
    """Test the RTDETR functionality within Ultralytics for detection tasks using specified model and data."""
    # Warning: must use imgsz=640 (note also add coma, spaces, fraction=0.25 args to test single-image training)
    run(f"yolo train {task} model={model} data={data} --imgsz= 160 epochs =1, cache = disk fraction=0.25")
    run(f"yolo predict {task} model={model} source={ASSETS / 'bus.jpg'} imgsz=160 save save_crop save_txt")


@pytest.mark.skipif(checks.IS_PYTHON_3_12, reason="MobileSAM with CLIP is not supported in Python 3.12")
def test_fastsam(task="segment", model=WEIGHTS_DIR / "FastSAM-s.pt", data="coco8-seg.yaml"):
    """Test FastSAM model for segmenting objects in images using various prompts within Ultralytics."""
    source = ASSETS / "bus.jpg"

    run(f"yolo segment val {task} model={model} data={data} imgsz=32")
    run(f"yolo segment predict model={model} source={source} imgsz=32 save save_crop save_txt")

    from ultralytics import FastSAM
    from ultralytics.models.fastsam import FastSAMPrompt
    from ultralytics.models.sam import Predictor

    # Create a FastSAM model
    sam_model = FastSAM(model)  # or FastSAM-x.pt

    # Run inference on an image
    for s in (source, Image.open(source)):
        everything_results = sam_model(s, device="cpu", retina_masks=True, imgsz=320, conf=0.4, iou=0.9)

        # Remove small regions
        new_masks, _ = Predictor.remove_small_regions(everything_results[0].masks.data, min_area=20)

        # Everything prompt
        prompt_process = FastSAMPrompt(s, everything_results, device="cpu")
        ann = prompt_process.everything_prompt()

        # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
        ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])

        # Text prompt
        ann = prompt_process.text_prompt(text="a photo of a dog")

        # Point prompt
        # Points default [[0,0]] [[x1,y1],[x2,y2]]
        # Point_label default [0] [1,0] 0:background, 1:foreground
        ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
        prompt_process.plot(annotations=ann, output="./")


def test_mobilesam():
    """Test MobileSAM segmentation with point prompts using Ultralytics."""
    from ultralytics import SAM

    # Load the model
    model = SAM(WEIGHTS_DIR / "mobile_sam.pt")

    # Source
    source = ASSETS / "zidane.jpg"

    # Predict a segment based on a point prompt
    model.predict(source, points=[900, 370], labels=[1])

    # Predict a segment based on a box prompt
    model.predict(source, bboxes=[439, 437, 524, 709])

    # Predict all
    # model(source)


# Slow Tests -----------------------------------------------------------------------------------------------------------
@pytest.mark.slow
@pytest.mark.parametrize("task,model,data", TASK_MODEL_DATA)
@pytest.mark.skipif(not CUDA_IS_AVAILABLE, reason="CUDA is not available")
@pytest.mark.skipif(CUDA_DEVICE_COUNT < 2, reason="DDP is not available")
def test_train_gpu(task, model, data):
    """Test YOLO training on GPU(s) for various tasks and models."""
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0")  # single GPU
    run(f"yolo train {task} model={model} data={data} imgsz=32 epochs=1 device=0,1")  # multi GPU
