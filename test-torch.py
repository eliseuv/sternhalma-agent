import torch

print("CUDA support:", torch.cuda.is_available())
print("CUDA devices count:", torch.cuda.device_count())
print("CUDA devices name:", torch.cuda.get_device_name(0))
