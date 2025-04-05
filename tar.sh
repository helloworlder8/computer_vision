#!/bin/bash

# 获取当前年份、月份和日期
current_date=$(date +%Y-%m-%d)

# 归档文件的路径和名称，包括年份、月份和日期
archive_file="../computer_vision-$current_date.tar.gz"

# 要归档的目录
source_directory="../computer_vision-11-14"

# 使用 tar 命令创建归档文件，同时排除 .pt 文件、checkpoints 和 runs 目录
tar --exclude="*.pt" --exclude="$source_directory/runs" --exclude="$source_directory/datasets" -czvf "$archive_file" -C "$source_directory" .

echo "Archive created: $archive_file"


# alss v8-ghost v6 v8-p6 v8t 