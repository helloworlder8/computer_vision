#!/bin/bash

# 获取当前年份、月份和日期
current_date=$(date +%Y-%m-%d)

# 归档文件的路径和名称，包括年份、月份和日期
archive_file="../ultralytics-8.2.60_$current_date.tar.gz"

# 要归档的目录
source_directory="../ultralytics-8.2.60"

# 使用 tar 命令创建归档文件，同时排除 .pt 文件、checkpoints 和 runs 目录
tar --exclude="*.pt" --exclude="$source_directory/checkpoints" --exclude="$source_directory/runs" -czvf "$archive_file" -C "$source_directory" .

echo "Archive created: $archive_file"
