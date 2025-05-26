import torch

a = torch.tensor([0], device="cuda", dtype=torch.float16)
b = a
c = a / b
print(c.to(torch.uint8))
