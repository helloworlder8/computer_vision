import os
import shutil
import re
# 定义源目录和目标目录
source_dir = 'runs/detect'
target_dir = 'plot'

# 创建目标目录，如果它不存在的话
if not os.path.exists(target_dir):
    os.makedirs(target_dir)

# 遍历源目录下的所有子目录
for root, dirs, files in os.walk(source_dir):
    for file in files:
        # 检查文件是否是我们要复制的 result.txt
        # if file == '*1.jpg':
        if re.match('.*526.jpg$', file):
        # if file == 'F1_curve.csv':
        # if file == 'best.pt':
            # parent_folder = root.split('/')[-2]  # 从倒数第二个'/'字符开始取
            parent_folder = os.path.basename(root)
            parent_folder = parent_folder + "occlusion.jpg"
            # 构建源文件完整路径
            source_file = os.path.join(root, file)
            new_file_name = f"{file}"
            # 为每个文件构建目标路径，这里我们保留相对子目录结构
            relative_path = os.path.relpath(root, source_dir)

            # 创建目标文件的目录（如果不存在）
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
            # 构建目标文件的完整路径
            target_file = os.path.join(target_dir, parent_folder)
            # 复制文件
            shutil.copy2(source_file, target_file) #需要完整路径
            print(f"Copied: {source_file} to {target_file}")

print("All 'result.txt' files have been copied to the 'plot' directory.")
