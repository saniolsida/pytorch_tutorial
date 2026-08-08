import torch
import numpy as np


# t = np.array([[[0, 1, 2], [3, 4, 5]], [[6, 7, 8], [9, 10, 11]]])
# ft = torch.FloatTensor(t)

# print(ft.view([-1, 3]))  # -1 = 자동 계산이라는 의미이다.

# print(ft.view([-1, 1, 3]))

# ft = torch.FloatTensor([[0], [1], [2]])
# print(ft.squeeze())
# print(ft.squeeze().shape) # squeeze() = 1인 차원을 없에준다.

# ft = torch.Tensor([0, 1, 2])

# print(ft.unsqueeze(0))
# print(ft.view(1, -1))

# print(ft.unsqueeze(1))
# print(ft.unsqueeze(-1))


# type casting
# lt = torch.LongTensor([1, 2, 3, 4])
# print(lt.float())
# bt = torch.ByteTensor([True, False, False, True])
# print(bt)
# print(bt.long())

# concatenate
# x = torch.FloatTensor([[1, 2], [3, 4]])
# y = torch.FloatTensor([[5, 6], [7, 8]])

# print(torch.cat([x, y], dim=0))
# print(torch.cat([x, y], dim=1))

# stack
# stack은 새로운 차원을 하나 만들어서 쌓는다.
# 반면 cat은 기존 차원 중 하나를 따라 이어 붙인다. 새로운 차원을 만들지 않는다.

# x = torch.FloatTensor([1, 4])
# y = torch.FloatTensor([2, 5])
# z = torch.FloatTensor([3, 6])

# print(torch.stack([x, y, z]))
# print(torch.stack([x, y, z], dim=1))

# print(torch.cat([x.unsqueeze(0), y.unsqueeze(0), z.unsqueeze(0)], dim=0))
# print(x.shape)
# print(x.unsqueeze(0))
# print(x.unsqueeze(1))

# x = torch.tensor([[1, 2], [3, 4]])

# print("-------------")
# print(x.unsqueeze(0))
# print("-------------")
# print(x.unsqueeze(1))
# print("-------------")
# print(x.unsqueeze(2))

# ones and zeros: 동일 크기의 배열을 만든다.

# x = torch.FloatTensor([[0, 1, 2], [2, 1, 0]])

# print(torch.ones_like(x))
# print(torch.zeros_like(x))

# in place operation

x = torch.FloatTensor([[1, 2], [3, 4]])

print(x.mul(2.0))
print(x)
print(x.mul_(2.0))  # x에 직접 데이터를 넣는다?
print(x)
