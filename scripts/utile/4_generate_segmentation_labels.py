from ultralytics import SAM

# Load a model
model = SAM("sam_b.pt")

# Segment with bounding box prompt
sam_results = model("ultralytics/assets/zidane.jpg")

# Segment with points prompt
# model("ultralytics/assets/zidane.jpg", points=[900, 370], labels=[1])

output_path = './1.txt'
segments = sam_results[0].masks.xyn  # noqa
# 为每个图像生成标注文件
with open(output_path, "w") as f:
    # 增量ID初始化
    incremental_id = 0
    # 遍历每个分割区域
    for i in range(len(segments)):
        s = segments[i]
        # 如果分割区域为空，则跳过
        if len(s) == 0:
            continue
        # 将分割区域坐标转换为字符串格式
        segment = map(str, segments[i].reshape(-1).tolist())
        # 写入标注信息，使用增量ID
        f.write(f"{incremental_id} " + " ".join(segment) + "\n")
        # 增量ID递增
        incremental_id += 1
