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
    DEFAULT_PARAM,
    DEFAULT_PARAM_DICT,
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
MODES = "train", "val", "predict", "export", "track", "benchmark"
TASKS = "detect", "segment", "classify", "pose", "obb"
TASK_TO_DATA_CFG = {
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

CLI_HELP_MSG = f"""
    Arguments received: {str(['yolo'] + sys.argv[1:])}. Ultralytics 'yolo' commands use the following syntax:
    """

# Define keys for arg type checks
CFG_FLOAT_KEYS = "warmup_epochs", "box", "cls", "dfl", "degrees", "shear", "time"
CFG_FRACTION_KEYS = (
    "dropout",
    "iou",
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
    "mosaic",
    "mixup",
    "copy_paste",
    "conf",
    "iou",
    "fraction",
)  # fraction floats 0.0 - 1.0
CFG_INT_KEYS = (
    "epochs",
    "patience",
    "batch",
    "workers",
    "seed",
    "close_mosaic",
    "mask_ratio",
    "max_det",
    "vid_stride",
    "line_width",
    "workspace",
    "nbs",
    "save_period",
)
CFG_BOOL_KEYS = (
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
)


def param_to_dict(cfg):

    if isinstance(cfg, (str, Path)):
        cfg = yaml_load(cfg)  # load dict
    elif isinstance(cfg, SimpleNamespace):
        cfg = vars(cfg)  # convert to dict
    return cfg







def creat_args(default_param: Union[str, Path, Dict, SimpleNamespace] = DEFAULT_PARAM_DICT, overrides: Dict = None):
    args = _merge_args(default_param, overrides)
    _special_handling(args)
    _type_and_value_checks(args)
    return IterableSimpleNamespace(**args)

def _merge_args(default_param, overrides):
    default_param = param_to_dict(default_param)
    if overrides:
        overrides = param_to_dict(overrides)
        if "save_dir" not in default_param:
            overrides.pop("save_dir", None)  # special override keys to ignore
        check_dict_alignment(default_param, overrides)
        args = {**default_param, **overrides}
    else:
        args = default_param
    return args

def _special_handling(args):
    for k in "project", "name":
        if k in args and isinstance(args[k], (int, float)):
            args[k] = str(args[k])
        
def _type_and_value_checks(args):
    for k, v in args.items():
        if v is not None:
            if k in CFG_FLOAT_KEYS and not isinstance(v, (int, float)):
                raise TypeError(
                    f"'{k}={v}' is of invalid type {type(v).__name__}. "
                    f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                )
            elif k in CFG_FRACTION_KEYS:
                if not isinstance(v, (int, float)):
                    raise TypeError(
                        f"'{k}={v}' is of invalid type {type(v).__name__}. "
                        f"Valid '{k}' types are int (i.e. '{k}=0') or float (i.e. '{k}=0.5')"
                    )
                if not (0.0 <= v <= 1.0):
                    raise ValueError(f"'{k}={v}' is an invalid value. " f"Valid '{k}' values are between 0.0 and 1.0.")
            elif k in CFG_INT_KEYS and not isinstance(v, int):
                raise TypeError(
                    f"'{k}={v}' is of invalid type {type(v).__name__}. " f"'{k}' must be an int (i.e. '{k}=8')"
                )
            elif k in CFG_BOOL_KEYS and not isinstance(v, bool):
                raise TypeError(
                    f"'{k}={v}' is of invalid type {type(v).__name__}. "
                    f"'{k}' must be a bool (i.e. '{k}=True' or '{k}=False')"
                )









def creat_save_dir(args, name=None):
    """Return save_dir as created from train/val/predict arguments."""

    if getattr(args, "save_dir", None): #告诉名称以名称为主
        save_dir = args.save_dir
    else:
        from ultralytics.utils.files import increment_path
        
        project = args.project or (ROOT.parent / "tests/tmp/runs" if TESTS_RUNNING else RUNS_DIR) / args.task_name #运行路径加任务名
        name = name or args.name or f"{args.mode}" #这种索引方式值得学习 穿名 参名 模式
        save_dir = increment_path(Path(project) / name, exist_ok=args.exist_ok if RANK in (-1, 0) else True)

    return Path(save_dir)


def _handle_deprecation(overrides):
    """Hardcoded function to handle deprecated config keys."""

    for key in overrides.copy().keys():
        if key == "boxes":
            deprecation_warn(key, "show_boxes")
            overrides["show_boxes"] = overrides.pop("boxes")
        if key == "hide_labels":
            deprecation_warn(key, "show_labels")
            overrides["show_labels"] = overrides.pop("hide_labels") == "False"
        if key == "hide_conf":
            deprecation_warn(key, "show_conf")
            overrides["show_conf"] = overrides.pop("hide_conf") == "False"
        if key == "line_thickness":
            deprecation_warn(key, "line_width")
            overrides["line_width"] = overrides.pop("line_thickness")

    return overrides


def check_dict_alignment(default_param: Dict, overrides: Dict, e=None):

    overrides = _handle_deprecation(overrides)
    default_keys, overrides_keys = (set(x.keys()) for x in (default_param, overrides)) #104 5
    mismatched = [k for k in overrides_keys if k not in default_keys] #overrides_keys有但是default_keys没有
    if mismatched: #[]才正常
        from difflib import get_close_matches

        string = ""
        for x in mismatched:
            matches = get_close_matches(x, default_keys)  # key list
            matches = [f"{k}={default_param[k]}" if default_param.get(k) is not None else k for k in matches]
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

    This function processes Ultralytics HUB CLI commands such as login and logout.
    It should be called when executing a script with arguments related to HUB authentication.

    Args:
        args (List[str]): A list of command line arguments

    Example:
        ```bash
        python my_script.py hub login your_api_key
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

    This function processes YOLO settings CLI commands such as reset.
    It should be called when executing a script with arguments related to YOLO settings management.

    Args:
        args (List[str]): A list of command line arguments for YOLO settings management.

    Example:
        ```bash
        python my_script.py yolo settings reset
        ```
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
    """Open the Ultralytics Explorer GUI."""
    checks.check_requirements("streamlit")
    LOGGER.info("💡 Loading Explorer dashboard...")
    subprocess.run(["streamlit", "run", ROOT / "data/explorer/gui/dash.py", "--server.maxMessageSize", "2048"])


def parse_key_value_pair(pair):
    """Parse one 'key=value' pair and return key and value."""
    k, v = pair.split("=", 1)  # split on first '=' sign
    k, v = k.strip(), v.strip()  # remove spaces
    assert v, f"missing '{k}' value"
    return k, smart_value(v)


def smart_value(v):
    """Convert a string to an underlying type such as int, float, bool, etc."""
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
    This function is the ultralytics package entrypoint, it's responsible for parsing the command line arguments passed
    to the package.

    This function allows for:
    - passing mandatory YOLO args as a list of strings
    - specifying the task_name to be performed, either 'detect', 'segment' or 'classify'
    - specifying the mode, either 'train', 'val', 'test', or 'predict'
    - running special modes like 'checks'
    - passing overrides to the package's configuration

    It uses the package's default cfg and initializes it using the passed overrides.
    Then it calls the CLI function with the composed cfg
    """
    args = (debug.split(" ") if debug else sys.argv)[1:]
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
    }
    full_args_dict = {**DEFAULT_PARAM_DICT, **{k: None for k in TASKS}, **{k: None for k in MODES}, **special}

    # Define common misuses of special commands, i.e. -h, -help, --help
    special.update({k[0]: v for k, v in special.items()})  # singular
    special.update({k[:-1]: v for k, v in special.items() if len(k) > 1 and k.endswith("s")})  # singular
    special = {**special, **{f"-{k}": v for k, v in special.items()}, **{f"--{k}": v for k, v in special.items()}}

    overrides = {}  # basic overrides, i.e. imgsz=320
    for a in merge_equals_args(args):  # merge spaces around '=' sign
        if a.startswith("--"):
            LOGGER.warning(f"WARNING ⚠️ '{a}' does not require leading dashes '--', updating to '{a[2:]}'.")
            a = a[2:]
        if a.endswith(","):
            LOGGER.warning(f"WARNING ⚠️ '{a}' does not require trailing comma ',', updating to '{a[:-1]}'.")
            a = a[:-1]
        if "=" in a:
            try:
                k, v = parse_key_value_pair(a)
                if k == "cfg" and v is not None:  # overrides.yaml passed
                    LOGGER.info(f"Overriding {DEFAULT_CFG_PATH} with {v}")
                    overrides = {k: val for k, val in yaml_load(checks.check_yaml(v)).items() if k != "cfg"}
                else:
                    overrides[k] = v
            except (NameError, SyntaxError, ValueError, AssertionError) as e:
                check_dict_alignment(full_args_dict, {a: ""}, e)

        elif a in TASKS:
            overrides["task_name"] = a
        elif a in MODES:
            overrides["mode"] = a
        elif a.lower() in special:
            special[a.lower()]()
            return
        elif a in DEFAULT_PARAM_DICT and isinstance(DEFAULT_PARAM_DICT[a], bool):
            overrides[a] = True  # auto-True for default bool args, i.e. 'yolo show' sets show=True
        elif a in DEFAULT_PARAM_DICT:
            raise SyntaxError(
                f"'{colorstr('red', 'bold', a)}' is a valid YOLO argument but is missing an '=' sign "
                f"to set its value, i.e. try '{a}={DEFAULT_PARAM_DICT[a]}'\n{CLI_HELP_MSG}"
            )
        else:
            check_dict_alignment(full_args_dict, {a: ""})

    # Check keys
    check_dict_alignment(full_args_dict, overrides)

    # Mode
    mode = overrides.get("mode")
    if mode is None:
        mode = DEFAULT_PARAM.mode or "predict"
        LOGGER.warning(f"WARNING ⚠️ 'mode' is missing. Valid modes are {MODES}. Using default 'mode={mode}'.")
    elif mode not in MODES:
        raise ValueError(f"Invalid 'mode={mode}'. Valid modes are {MODES}.\n{CLI_HELP_MSG}")

    # Task
    task_name = overrides.pop("task_name", None)
    if task_name:
        if task_name not in TASKS:
            raise ValueError(f"Invalid 'task_name={task_name}'. Valid tasks are {TASKS}.\n{CLI_HELP_MSG}")
        if "model" not in overrides:
            overrides["model"] = TASK2MODEL[task_name]

    # Model
    model = overrides.pop("model", DEFAULT_PARAM.model)
    if model is None:
        model = "yolov8n.pt"
        LOGGER.warning(f"WARNING ⚠️ 'model' is missing. Using default 'model={model}'.")
    overrides["model"] = model
    stem = Path(model).stem.lower()
    if "rtdetr" in stem:  # guess architecture
        from ultralytics import RTDETR

        model = RTDETR(model)  # no task_name argument
    elif "fastsam" in stem:
        from ultralytics import FastSAM

        model = FastSAM(model)
    elif "sam" in stem:
        from ultralytics import SAM

        model = SAM(model)
    else:
        from ultralytics import YOLO

        model = YOLO(model, task_name=task_name)
    if isinstance(overrides.get("pretrained"), str):
        model.load(overrides["pretrained"])

    # Task Update
    if task_name != model.task_name:
        if task_name:
            LOGGER.warning(
                f"WARNING ⚠️ conflicting 'task_name={task_name}' passed with 'task_name={model.task_name}' model. "
                f"Ignoring 'task_name={task_name}' and updating to 'task_name={model.task_name}' to match model."
            )
        task_name = model.task_name

    # Mode
    if mode in ("predict", "track") and "source" not in overrides:
        overrides["source"] = DEFAULT_PARAM.source or ASSETS
        LOGGER.warning(f"WARNING ⚠️ 'source' is missing. Using default 'source={overrides['source']}'.")
    elif mode in ("train", "val"):
        if "data" not in overrides and "resume" not in overrides:
            overrides["data"] = DEFAULT_PARAM.data or TASK_TO_DATA_CFG.get(task_name or DEFAULT_PARAM.task_name, DEFAULT_PARAM.data)
            LOGGER.warning(f"WARNING ⚠️ 'data' is missing. Using default 'data={overrides['data']}'.")
    elif mode == "export":
        if "format" not in overrides:
            overrides["format"] = DEFAULT_PARAM.format or "torchscript"
            LOGGER.warning(f"WARNING ⚠️ 'format' is missing. Using default 'format={overrides['format']}'.")

    # Run command in python
    getattr(model, mode)(**overrides)  # default args from model

    # Show help
    LOGGER.info(f"💡 Learn more at https://docs.ultralytics.com/modes/{mode}")


# Special modes --------------------------------------------------------------------------------------------------------
def copy_default_cfg():
    """Copy and create a new default configuration file with '_copy' appended to its name."""
    new_file = Path.cwd() / DEFAULT_CFG_PATH.name.replace(".yaml", "_copy.yaml")
    shutil.copy2(DEFAULT_CFG_PATH, new_file)
    LOGGER.info(
        f"{DEFAULT_CFG_PATH} copied to {new_file}\n"
        f"Example YOLO command with this new overrides cfg:\n    yolo cfg='{new_file}' imgsz=320 batch=8"
    )


if __name__ == "__main__":
    # Example: entrypoint(debug='yolo predict model=yolov8n.pt')
    entrypoint(debug="")
