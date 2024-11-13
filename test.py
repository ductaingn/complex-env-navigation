import torch

# Kiểm tra xem có GPU nào khả dụng không
if torch.cuda.is_available():
    print("GPU is available.")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {num_gpus}")

    # In ra thông tin từng GPU
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"  - Memory Allocated: {torch.cuda.memory_allocated(i) / 1024 ** 2:.2f} MB")
        print(f"  - Memory Cached: {torch.cuda.memory_reserved(i) / 1024 ** 2:.2f} MB")
else:
    print("GPU is not available.")
    
if torch.cuda.is_available():
    x = torch.rand(1000, 1000, device="cuda")
    y = torch.rand(1000, 1000, device="cuda")
    z = x @ y
    print("Computation on GPU is successful.")

