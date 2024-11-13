#!/bin/bash

# 定义要移动的目录
directories=(
    "/home/easyits/ai/samba/wp0702/roadrisk12/"
    "/home/easyits/ai/samba/wp0702/roadrisk3/"
    "/home/easyits/ai/samba/wp0702/roadrisk4/"
    "/home/easyits/ai/samba/wp0702/roadrisk5/"
    "/home/easyits/ai/samba/wp0702/roadrisk6/"
    "/home/easyits/ai/samba/wp0702/roadrisk7/"
)

# 创建datasets文件夹（如果不存在的话）
mkdir -p datasets

# 循环移动每个目录
for dir in "${directories[@]}"; do
    if mv "$dir" "datasets/"; then
        echo "成功移动: $dir"
    else
        echo "移动失败: $dir"
    fi
done
