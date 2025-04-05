import torch
import subprocess

def get_free_gpu(start_index=1):
    # 使用nvidia-smi命令获取GPU利用率信息
    result = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"])
    gpu_usages = result.decode("utf-8").strip().split("\n")
    
    # 从指定的起始索引开始查找完全空闲的GPU
    num_gpus = len(gpu_usages)
    for i in range(start_index, num_gpus):
        if int(gpu_usages[i]) == 0:
            return i
    
    # 如果没有找到空闲的GPU，从头开始查找
    for i in range(start_index):
        if int(gpu_usages[i]) == 0:
            return i
    
    return '0'

gpu_id = get_free_gpu(start_index=1)
if gpu_id is not None:
    device = torch.device(f"cuda:{gpu_id}")
    print(f"Using GPU: cuda:{gpu_id}")
else:
    device = torch.device("cpu")
    print("No free GPU found, using CPU")
    
    
    # 初始化根据语义写成几个内部的成员函数，使其简洁明了，但不改变整体逻辑