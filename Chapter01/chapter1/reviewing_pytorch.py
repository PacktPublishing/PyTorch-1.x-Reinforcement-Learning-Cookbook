'''
Source codes for PyTorch 1.0 Reinforcement Learning (Packt Publishing)
Chapter 1: Getting started with reinforcement learning and PyTorch
Author: Yuxi (Hayden) Liu
'''

import torch


x = torch.rand(3, 4)
print(x)

x = torch.rand(3, 4, dtype=torch.double)
print(x)

x = torch.zeros(3, 4)
print(x)

x = torch.ones(3, 4)
print(x)
print(x.size())
x_reshaped = x.view(2, 6)
print(x_reshaped)

x1 = torch.tensor(3)
print(x1)
x2 = torch.tensor([14.2, 3, 4])
print(x2)
x3 = torch.tensor([[3, 4, 6], [2, 1.0, 5]])
print(x3)

print(x2[1])
print(x3[1, 0])
print(x3[:, 1:])
print(x1.item())


print(x3.numpy())

import numpy as np
x_np = np.ones(3)
x_torch = torch.from_numpy(x_np)
print(x_torch)
print(x_torch.float())


x4 = torch.tensor([[1, 0, 0], [0, 1.0, 0]])
print(x3 + x4)
print(torch.add(x3, x4))


x3.add_(x4)
print(x3)

