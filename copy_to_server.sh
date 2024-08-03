#!/bin/bash

# Remote server details
remote_user="gcsx"
remote_host="10.20.4.168"

# Declare an associative array where
#   key = local file path
#   value = remote directory path
declare -A FILES_TO_COPY=(
    ["ultralytics/nn/common/modules.py"]="/home/gcsx/ANG/ultralytics-24-03-30/ultralytics/nn/common/modules.py" #基本模块 1 
    ["ultralytics/nn/common/__init__.py"]="/home/gcsx/ANG/ultralytics-24-03-30/ultralytics/nn/common/__init__.py" #给外部调用 2
    ["ultralytics/nn/parse_model.py"]="/home/gcsx/ANG/ultralytics-24-03-30/ultralytics/nn/parse_model.py" #搭建模型 2
    ["train_server.py"]="/home/gcsx/ANG/ultralytics-24-03-30/train_server.py" #运行
    

    # Add more files and their target directories as needed
    # ["local_path"]="remote_directory"
)

# Iterate over the associative array
for local_file in "${!FILES_TO_COPY[@]}"; do
    # Get the corresponding remote directory for the current file
    remote_dir="${FILES_TO_COPY[$local_file]}"

    # Check if the local file exists
    if [ ! -f "$local_file" ]; then
        echo "File $local_file does not exist."
        continue
    fi

    # Copy the file to the remote directory

    # 在 scp 命令之前打印出完整命令
    echo "Executing: ${local_file}" "${remote_user}@${remote_host}:${remote_dir}"

    scp "${local_file}" "${remote_user}@${remote_host}:${remote_dir}"

    # Check if SCP succeeded
    if [ $? -eq 0 ]; then
        echo "Successfully"
    else
        echo "Failed"
    fi
done

echo "Done copying files."


