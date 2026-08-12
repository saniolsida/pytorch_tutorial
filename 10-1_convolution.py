# Convolution: input X filter = output
# stride: filter를 한번에 얼마나 이동할 것인가.
# padding: input이미지 테두리에 padding 크기만큼 띄가 둘러진다.

# ex1)
# input image size = 227*227
# filter size = 11*11
# stride = 4, padding = 0
# 227 - 11 = 216/4 + 1 = 55

# ex2)
# input image size = 64*64
# filter size = 7*7
# stride = 2, padding = 0
# 64-7 = 57/2 = 28 + 1 = 29

# ex3)
# input image size = 32*64
# filter size = 5*5
# stride = 1, padding = 0
# (32-5),(64-5) = (27,59) + 1 = 28X60

import torch
import torch.nn as nn

# conv = nn.Conv2d(1, 1, 7, stride=2, padding=0)
# inputs = torch.Tensor(1, 1, 64, 64)
# out = conv(inputs)
# print(out.shape)

# conv = nn.Conv2d(1, 1, 5, stride=1, padding=2)
# inputs = torch.Tensor(1, 1, 32, 32)
# out = conv(inputs)
# print(out.shape)

# pooling: 이미지 사이즈 축소 가능. Max Pooling = 2X2 안에서 가장 큰 값을 모아서 사용
# Average Pooling: 2X2 안에서 평균을 계산하여 출력

input = torch.Tensor(1, 1, 28, 28)
conv1 = nn.Conv2d(1, 5, 5)
pool = nn.MaxPool2d(2)
out = conv1(input)
out2 = pool(out)
print(out.size())
print(out2.size())
