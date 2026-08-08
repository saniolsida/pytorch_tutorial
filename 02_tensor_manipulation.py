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
