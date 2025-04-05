# Ultralytics YOLO 🚀, AGPL-3.0 license

import contextlib
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Dict, List, Union

from ultralytics.utils import (
    ASSETS,
    DEFAULT_CFG,
    DEFAULT_CFG_DICT,
    DEFAULT_CFG_PATH,
    LOGGER,
    RANK,
    ROOT,
    RUNS_DIR,
    SETTINGS,
    SETTINGS_YAML,
    TESTS_RUNNING,
    IterableSimpleNamespace,
    __version__,
    checks,
    colorstr,
    deprecation_warn,
    yaml_load,
    yaml_print,
)

# Define valid tasks and modes
MODES = {"train", "val", "predict", "export", "track", "benchmark"}
TASKS = {"detect", "segment", "classify", "pose", "obb"}
TASK2DATA = {
    "detect": "coco8.yaml",
    "segment": "coco8-seg.yaml",
    "classify": "imagenet10",
    "pose": "coco8-pose.yaml",
    "obb": "dota8.yaml",
}
TASK2MODEL = {
    "detect": "yolov8n.pt",
    "segment": "yolov8n-seg.pt",
    "classify": "yolov8n-cls.pt",
    "pose": "yolov8n-pose.pt",
    "obb": "yolov8n-obb.pt",
}
TASK2METRIC = {
    "detect": "metrics/mAP50-95(B)",
    "segment": "metrics/mAP50-95(M)",
    "classify": "metrics/accuracy_top1",
    "pose": "metrics/mAP50-95(P)",
    "obb": "metrics/mAP50-95(B)",
}
MODELS = {TASK2MODEL[task] for task in TASKS}

ARGV = sys.argv or ["", ""]  # sometimes sys.argv = []
CLI_HELP_MSG = f"""
    Arguments received: {str(['yolo'] + ARGV[1:])}. Ultralytics 'yolo' commands use the following syntax:

        yolo TASK MODE ARGS

        Where   TASK (optional) is one of {TASKS}
                MODE (required) is one of {MODES}
                ARGS (optional) are any number of custom 'arg=value' pairs like 'imgsz=320' that override defaults.
                    See all ARGS at https://docs.ultralytics.com/usage/cfg or with 'yolo cfg'

    1. Train a detection model for 10 epochs with an initial learning_rate of 0.01
        yolo train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01

    2. Predict a YouTube video using a pretrained segmentation model at image size 320:
        yolo predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320

    3. Val a pretrained detection model at batch-size 1 and image size 640:
        yolo val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640

    4. Export a YOLOv8n classification model to ONNX format at image size 224 by 128 (no TASK required)
        yolo export model=yolov8n-cls.pt format=onnx imgsz=224,128

    5. Explore your datasets using semantic search and SQL with a simple GUI powered by Ultralytics Explorer API
        yolo explorer
    
    6. Streamlit real-time object detection on your webcam with Ultralytics YOLOv8
        yolo streamlit-predict
        
    7. Run special commands:
        yolo help
        yolo checks
        yolo version
        yolo settings
        yolo copy-cfg
        yolo cfg

    Docs: https://docs.ultralytics.com
    Community: https://community.ultralytics.com
    GitHub: https://github.com/ultralytics/ultralytics
    """

# Define keys for arg type checks
CFG_FLOAT_KEYS = {  # integer or float arguments, i.e. x=2 and x=2.0
    "warmup_epochs",
    "box",
    "cls",
    "dfl",
    "degrees",
    "shear",
    "time",
    "workspace",
    "batch",
}
CFG_FRACTION_KEYS = {  # fractional float arguments with 0.0<=values<=1.0
    "dropout",
    "lr0",
    "lrf",
    "momentum",
    "weight_decay",
    "warmup_momentum",
    "warmup_bias_lr",
    "label_smoothing",
    "hsv_h",
    "hsv_s",
    "hsv_v",
    "translate",
    "scale",
    "perspective",
    "flipud",
    "fliplr",
    "bgr",
    "mosaic",
    "mixup",
    "copy_paste",
    "conf",
    "iou",
    "fraction",
}
CFG_INT_KEYS = {  # integer-only arguments
    "epochs",
    "patience",
    "workers",
    "seed",
    "close_mosaic",
    "mask_ratio",
    "max_det",
    "vid_stride",
    "line_width",
    "nbs",
    "save_period",
}
CFG_BOOL_KEYS = {  # boolean-only arguments
    "save",
    "exist_ok",
    "verbose",
    "deterministic",
    "single_cls",
    "rect",
    "cos_lr",
    "overlap_mask",
    "val",
    "save_json",
    "save_hybrid",
    "half",
    "dnn",
    "plots",
    "show",
    "save_txt",
    "save_conf",
    "save_crop",
    "save_frames",
    "show_labels",
    "show_conf",
    "visualize",
    "augment",
    "agnostic_nms",
    "retina_masks",
    "show_boxes",
    "keras",
    "optimize",
    "int8",
    "dynamic",
    "simplify",
    "nms",
    "profile",
    "multi_scale",
}


def cfg2dict(cfg):
    """
    Convert a configuration object to a dictionary, whether it is a file path, a string, or a SimpleNamespace object.

    Args:
        cfg (str | Path | dict | SimpleNamespace): Configuration object to be converted to a dictionary. This may be a
            path to a configuration file, a dictionary, or a SimpleNamespace object.

    Returns:
        (dict): Configuration object in dictionary format.

    Example:
        ```python
        from ultralytics.cfg import cfg2dict
        from types import SimpleNamespace

        # Example usage with a file path
        config_dict = cfg2dict('config.yaml')

        # Example usage with a SimpleNamespace
        config_sn = SimpleNamespace(param1='value1', param2='value2')
        config_dict = cfg2dict(config_sn)

        # Example usage with a dictionary (returns the same dictionary)
        config_dict = cfg2dict({'param1': 'value1', 'param2': 'value2'})
        ```

    Notes:
        - If `cfg` is a path or a string, it will be loaded as YAML and converted to a dictionary.
        - If `cfg` is a SimpleNamespace object, it will be converted to a dictionary using `vars()`.
    """
    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg


def get_args(default_cfg: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_CFG_DICT, overrides: Dict = None):


    args_dict = cfg2dict(default_cfg) #迭代器转为字典形式

    # Merge overrides
    if overrides:
        overrides_dict = cfg2dict(overrides)
        if "save_dir" not in args_dict:
            overrides_dict.pop("save_dir", None)  # special override keys to ignore
        check_dict_alignment(args_dict, overrides_dict)
        args_dict = {**args_dict, **overrides_dict}  # merge cfg and overrides dicts (prefer overrides)



    # Type and Value checks
    check_args_type_value(args_dict)

    # Return instance
    return IterableSimpleNamespace(**args_dict)


def check_args_type_value(args_dict, hard=True):
    """Validate and adjust Ultralytics configuration argument types and values."""
    # Convert numeric 'project' and 'name' arguments to strings
    for key in ["project", "name"]:
        if isinstance(args_dict.get(key), (int, float)):
            args_dict[key] = str(args_dict[key])

    # Assign 'model' to 'name' if 'name' is set to "model"
    if args_dict.get("name") == "model":
        args_dict["name"] = args_dict.get("model", "").split(".")[0]
        LOGGER.warning(f"WARNING ⚠️ 'name=model' automatically updated to 'name={args_dict['name']}'.")

    # Validate and convert argument types as per configuration
    for key, value in args_dict.items():
        if value is None:
            continue

        if key in CFG_FLOAT_KEYS:
            if not isinstance(value, (int, float)):
                if hard:
                    raise TypeError(f"'{key}={value}' must be an int or float.")
                args_dict[key] = float(value)

        elif key in CFG_FRACTION_KEYS:
            if not isinstance(value, (int, float)):
                if hard:
                    raise TypeError(f"'{key}={value}' must be an int or float.")
                args_dict[key] = value = float(value)
            if not (0.0 <= value <= 1.0):
                raise ValueError(f"'{key}={value}' must be between 0.0 and 1.0.")

        elif key in CFG_INT_KEYS:
            if not isinstance(value, int):
                if hard:
                    raise TypeError(f"'{key}={value}' must be an int.")
                args_dict[key] = int(value)

        elif key in CFG_BOOL_KEYS:
            if not isinstance(value, bool):
                if hard:
                    raise TypeError(f"'{key}={value}' must be a bool.")
                args_dict[key] = bool(value)



from pathlib import Path
from ultralytics.utils.files import increment_path

def get_save_dir(args, name=None):
    """Returns the directory path for saving outputs, derived from arguments or default settings."""

    # 使用 `args.save_dir` 优先保存路径
    if getattr(args, "save_dir", None):
        return Path(args.save_dir)

    # 根据任务和项目设置路径
    project = args.project or ((ROOT.parent / "tests/tmp/runs") if TESTS_RUNNING else RUNS_DIR) / args.task
    name = name or args.name or args.mode  # 设置保存目录的名称
    exist_ok = args.exist_ok if RANK in {-1, 0} else True  # 是否覆盖已有路径

    # 使用 `increment_path` 递增路径，确保不会覆盖已存在的文件夹
    return increment_path(Path(project) / name, exist_ok=exist_ok)



def _deprecation_warn(overrides_dict):
    """Handles deprecated configuration keys by mapping them to current equivalents with deprecation warnings."""

    for key in overrides_dict.copy().keys():
        if key == "boxes":
            deprecation_warn(key, "show_boxes")
            overrides_dict["show_boxes"] = overrides_dict.pop("boxes")
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")
            overrides_dict["show_labels"] = overrides_dict.pop("hide_labels") == "False"
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")
            overrides_dict["show_conf"] = overrides_dict.pop("hide_conf") == "False"
        if key == "line_thickness":
            deprecation_warn(key, "line_width")
            overrides_dict["line_width"] = overrides_dict.pop("line_thickness")

    return overrides_dict

# check_dict_alignment(default_cfg_dict, overrides_dict)
def check_dict_alignment(default_cfg_dict: Dict, overrides_dict: Dict, e=None):

    overrides_dict = _deprecation_warn(overrides_dict)
    default_keys, overrides_keys = (set(x.keys()) for x in (default_cfg_dict, overrides_dict))
    mismatched = [k for k in overrides_keys if k not in default_keys]
    if mismatched:
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, default_keys)  # key list
            matches = [f"{k}={default_cfg_dict[k]}" if default_cfg_dict.get(k) is not None else k for k in matches]
            match_str = f"Similar arguments are i.e. {matches}." if matches else ""
            string += f"'{colorstr('red', 'bold', x)}' is not a valid YOLO argument. {match_str}\n"
        raise SyntaxError(string + CLI_HELP_MSG) from e


def merge_equals_args(args: List[str]) -> List[str]:
    """
    Merges arguments around isolated '=' args in a list of strings. The function considers cases where the first
    argument ends with '=' or the second starts with '=', as well as when the middle one is an equals sign.

    Args:
        args (List[str]): A list of strings where each element is an argument.

    Returns:
        (List[str]): A list of strings where the arguments around isolated '=' are merged.

    Example:
        The function modifies the argument list as follows:
        ```python
        args = ["arg1", "=", "value"]
        new_args = merge_equals_args(args)
        print(new_args)  # Output: ["arg1=value"]

        args = ["arg1=", "value"]
        new_args = merge_equals_args(args)
        print(new_args)  # Output: ["arg1=value"]

        args = ["arg1", "=value"]
        new_args = merge_equals_args(args)
        print(new_args)  # Output: ["arg1=value"]
        ```
    """
    new_args = []
    for i, arg in enumerate(args):
        if arg == "=" and 0 < i < len(args) - 1:  # merge ['arg', '=', 'val']
            new_args[-1] += f"={args[i + 1]}"
            del args[i + 1]
        elif arg.endswith("=") and i < len(args) - 1 and "=" not in args[i + 1]:  # merge ['arg=', 'val']
            new_args.append(f"{arg}{args[i + 1]}")
            del args[i + 1]
        elif arg.startswith("=") and i > 0:  # merge ['arg', '=val']
            new_args[-1] += arg
        else:
            new_args.append(arg)
    return new_args


def handle_yolo_hub(args: List[str]) -> None:
    """
    Handle Ultralytics HUB command-line interface (CLI) commands.

    This function processes Ultralytics HUB CLI commands such as login and logout. It should be called when executing
    a script with arguments related to HUB authentication.

    Args:
        args (List[str]): A list of command line arguments.

    Returns:
        None

    Example:
        ```bash
        yolo hub login YOUR_API_KEY
        ```
    """
    from ultralytics import hub

    if args[0] == "login":
        key = args[1] if len(args) > 1 else ""
        # Log in to Ultralytics HUB using the provided API key
        hub.login(key)
    elif args[0] == "logout":
        # Log out from Ultralytics HUB
        hub.logout()


def handle_yolo_settings(args: List[str]) -> None:
    """
    Handle YOLO settings command-line interface (CLI) commands.

    This function processes YOLO settings CLI commands such as reset. It should be called when executing a script with
    arguments related to YOLO settings management.

    Args:
        args (List[str]): A list of command line arguments for YOLO settings management.

    Returns:
        None

    Example:
        ```bash
        yolo settings reset
        ```

    Notes:
        For more information on handling YOLO settings, visit:
        https://docs.ultralytics.com/quickstart/#ultralytics-settings
    """
    url = "https://docs.ultralytics.com/quickstart/#ultralytics-settings"  # help URL
    try:
        if any(args):
            if args[0] == "reset":
                SETTINGS_YAML.unlink()  # delete the settings file
                SETTINGS.reset()  # create new settings
                LOGGER.info("Settings reset successfully")  # inform the user that settings have been reset
            else:  # save a new setting
                new = dict(parse_key_value_pair(a) for a in args)
                check_dict_alignment(SETTINGS, new)
                SETTINGS.update(new)

        LOGGER.info(f"💡 Learn about settings at {url}")
        yaml_print(SETTINGS_YAML)  # print the current settings
    except Exception as e:
        LOGGER.warning(f"WARNING ⚠️ settings error: '{e}'. Please see {url} for help.")


def handle_explorer():
    """Open the Ultralytics Explorer GUI for dataset exploration and analysis."""
    checks.check_requirements("streamlit>=1.29.0")
    LOGGER.info("💡 Loading Explorer dashboard...")
    subprocess.run(["streamlit", "run", ROOT / "data/explorer/gui/dash.py", "--server.maxMessageSize", "2048"])


def handle_streamlit_inference():
    """Open the Ultralytics Live Inference streamlit app for real time object detection."""
    checks.check_requirements("streamlit>=1.29.0")
    LOGGER.info("💡 Loading Ultralytics Live Inference app...")
    subprocess.run(["streamlit", "run", ROOT / "solutions/streamlit_inference.py", "--server.headless", "true"])


def parse_key_value_pair(pair):
    """Parse one 'key=value' pair and return key and value."""
    k, v = pair.split("=", 1)  # split on first '=' sign
    k, v = k.strip(), v.strip()  # remove spaces
    assert v, f"missing '{k}' value"
    return k, smart_value(v)


def smart_value(v):
    """Convert a string to its appropriate type (int, float, bool, None, etc.)."""
    v_lower = v.lower()
    if v_lower == "none":
        return None
    elif v_lower == "true":
        return True
    elif v_lower == "false":
        return False
    else:
        with contextlib.suppress(Exception):
            return eval(v)
        return v


def entrypoint(debug=""):
    """
    Ultralytics entrypoint function for parsing and executing command-line arguments.

    This function serves as the main entry point for the Ultralytics CLI, parsing  command-line arguments and
    executing the corresponding tasks such as training, validation, prediction, exporting models, and more.

    Args:
        debug (str, optional): Space-separated string of command-line arguments for debugging purposes. Default is "".

    Returns:
        (None): This function does not return any value.

    Notes:
        - For a list of all available commands and their arguments, see the provided help messages and the Ultralytics
          documentation at https://docs.ultralytics.com.
        - If no arguments are passed, the function will display the usage help message.

    Example:
        ```python
        # Train a detection model for 10 epochs with an initial learning_rate of 0.01
        entrypoint("train data=coco8.yaml model=yolov8n.pt epochs=10 lr0=0.01")

        # Predict a YouTube video using a pretrained segmentation model at image size 320
        entrypoint("predict model=yolov8n-seg.pt source='https://youtu.be/LNwODJXcvt4' imgsz=320")

        # Validate a pretrained detection model at batch-size 1 and image size 640
        entrypoint("val model=yolov8n.pt data=coco8.yaml batch=1 imgsz=640")
        ```
    """
    args = (debug.split(" ") if debug else ARGV)[1:]
    if not args:  # no arguments passed
        LOGGER.info(CLI_HELP_MSG)
        return

    special = {
        "help": lambda: LOGGER.info(CLI_HELP_MSG),
        "checks": checks.collect_system_info,
        "version": lambda: LOGGER.info(__version__),
        "settings": lambda: handle_yolo_settings(args[1:]),
        "cfg": lambda: yaml_print(DEFAULT_CFG_PATH),
        "hub": lambda: handle_yolo_hub(args[1:]),
        "login": lambda: handle_yolo_hub(args),
        "copy-cfg": copy_default_cfg,
        "explorer": lambda: handle_explorer(),
        "streamlit-predict": lambda: handle_streamlit_inference(),
    }
    full_args_dict = {**DEFAULT_CFG_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    # Define common misuses of special commands, i.e. -h, -help, --help
    special.update({k[0]: v for k, v in special.items()})  # singular
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})  # singular
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}  # basic overrides, i.e. imgsz=320
    for a in merge_equals_args(args):  # merge spaces around '=' sign
        if a.startswith("--"):
            LOGGER.warning(f"WARNING ⚠️ argument '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"WARNING ⚠️ argument '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:  # custom.yaml passed
                    LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in yaml_load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_CFG_DICT and isinstance(DEFAULT_CFG_DICT[a], bool):
            overrides[a] = True  # auto-True for default bool args, i.e. 'yolo show' sets show=True
        elif a in DEFAULT_CFG_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={DEFAULT_CFG_DICT[a]}'\n{CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    # Check keys
    check_dict_alignment(full_args_dict, overrides)

    # Mode
    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_CFG.mode or "predict"
        LOGGER.warning(f"WARNING ⚠️ 'mode' argument is missing. Valid modes are {MODES}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {MODES}.\n{CLI_HELP_MSG}")

    # Task
    task = overrides.pop("task", None)
    if task:
        if task not in TASKS:
            raise ValueError(f"Invalid 'task={task}'. Valid tasks are {TASKS}.\n{CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task]

    # Model
    model = overrides.pop("model_name", DEFAULT_CFG.model_name)
    if model is None:
        model = "yolov8n.pt"
        LOGGER.warning(f"WARNING ⚠️ 'model' argument is missing. Using default 'model={model}'.")
    overrides["model"] = model
    stem = Path(model).stem.lower()
    if "rtdetr" in stem:  # guess architecture
        from ultralytics import RTDETR

        model = RTDETR(model)  # no task argument
    elif "fastsam" in stem:
        from ultralytics import FastSAM

        model = FastSAM(model)
    elif "sam" in stem:
        from ultralytics import SAM

        model = SAM(model)
    else:
        from ultralytics import YOLO

        model = YOLO(model, task=task)
    if isinstance(overrides.get("pretrained"), str):
        model.load(overrides["pretrained"])

    # Task Update
    if task != model.task:
        if task:
            LOGGER.warning(
                f"WARNING ⚠️ conflicting 'task={task}' passed with 'task={model.task}' model. "
                f"Ignoring 'task={task}' and updating to 'task={model.task}' to match model."
            )
        task = model.task

    # Mode
    if mode in {"predict", "track"} and "source" not in overrides:
        overrides["source"] = DEFAULT_CFG.source or ASSETS
        LOGGER.warning(f"WARNING ⚠️ 'source' argument is missing. Using default 'source={overrides['source']}'.")
    elif mode in {"train", "val"}:
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_CFG.data or TASK2DATA.get(task or DEFAULT_CFG.task, DEFAULT_CFG.data)
            LOGGER.warning(f"WARNING ⚠️ 'data' argument is missing. Using default 'data={overrides['data']}'.")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_CFG.format or "torchscript"
            LOGGER.warning(f"WARNING ⚠️ 'format' argument is missing. Using default 'format={overrides['format']}'.")

    # Run command in python
    getattr(model, mode)(**overrides)  # default args from model

    # Show help
    LOGGER.info(f"💡 Learn more at https://docs.ultralytics.com/modes/{mode}")


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_cfg():
    """Copy and create a new default configuration file with '_copy' appended to its name, providing usage example."""
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} copied to {new_file}\n"
        f"Example YOLO command with this new custom cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8"
    )


if __name__ == "__main__":
    # Example: entrypoint(debug='yolo predict model=yolov8n.pt')
    entrypoint(debug="")
