#!/bin/bash

# 获取当前年份、月份和日期
current_date=$(date +%Y-%m-%d)

# 归档文件的路径和名称，包括年份、月份和日期
archive_file="../论文2/computer_version_$current_date.tar.gz"

# 要归档的目录
source_directory="../computer_version"

# 使用tar命令创建归档文件
tar -czvf "$archive_file" "$source_directory"

echo "Archive created: $archive_file"