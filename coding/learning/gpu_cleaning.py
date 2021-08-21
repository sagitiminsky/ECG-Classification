import torch
from GPUtil import showUtilization as gpu_usage
from numba import cuda

def show_gpu_usage():
    print("Curent Usage")
    gpu_usage()

def free_gpu_cache():
    print("Initial GPU Usage")
    gpu_usage()

    torch.cuda.empty_cache()

    cuda.select_device(0)
    cuda.close()
    cuda.select_device(0)

    print("GPU Usage after emptying the cache")
    gpu_usage()

free_gpu_cache()
show_gpu_usage()