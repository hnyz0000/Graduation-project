import torch
print(torch.cuda.is_available())

print("当前可用的GPU数量: ", torch.cuda.device_count())
