#!/bin/bash
code ../computer_version -r ultralytics/nn/tasks_model.py \
ultralytics/cfg_yaml/test_model_yaml/ShuffleNet_24_04_04.3_lightcodattention.yaml \
script/train.py
# def parse_model(model_dict, ch, verbose=True):