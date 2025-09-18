import torch
torch.cuda.init()
print(torch.cuda.is_available())
print(torch.cuda.device_count())
x = torch.rand(1).to("cuda:1")

print(x)
