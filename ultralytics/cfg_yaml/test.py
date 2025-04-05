import yaml
import sys
yaml_file = "/home/easyits/ang/computer_vision-11-14/ultralytics/cfg_yaml/default.yaml"
def load_config_from_yaml(yaml_file):
    """从 YAML 文件加载配置"""
    with open(yaml_file, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config


config = load_config_from_yaml(yaml_file)