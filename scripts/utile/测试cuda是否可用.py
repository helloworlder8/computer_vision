import torch

def check_cuda_availability():
    if torch.cuda.is_available():
        print("CUDA is available. You can use GPU for computations.")
        print(f"Number of available GPUs: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. You will be using CPU for computations.")

if __name__ == "__main__":
    check_cuda_availability()
