import numpy as np
import torch

# t = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0], [0.0, 1.0, 2.0]])
# print(t)

# print("Rank of t: ", t.ndim)
# print("Shape of t: ", t.shape)

# t = torch.FloatTensor([0.0, 1.0, 1.0, 1.0, 1.0, 1.0])

# print(t.dim())
# print(t.shape)
# print(t.size())
# print(t[2:5])

# print(t)

# t = torch.FloatTensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]])

# print(t[:, 1])
# print(t[:, :-1])
# print(t[1:, 1:])

# m1 = torch.FloatTensor([[3, 3]])
# m2 = torch.FloatTensor([[2, 2]])
# print(m1 + m2)

# m1 = torch.FloatTensor([[3]])
# m2 = torch.FloatTensor([[2, 5]])
# print(m1 + m2)

# m1 = torch.FloatTensor([[2, 5], [3, 4]])
# m2 = torch.FloatTensor([[3], [6]])
# print(m1 + m2)
# print(m2)
# print(m1 @ m2)
# print(m2 * m1)

# t = torch.FloatTensor([1, 2])
# print(t.mean())

# t = torch.LongTensor([1, 2])
# try:
#     print(t.mean())
# except Exception as exc:
#     print(exc)

t = torch.FloatTensor([[1, 2], [3, 4]])
# print(t)

# print(t.mean())
# print(t.mean(dim=0))
# print(t.mean(dim=1))
# print(t.mean(dim=-1))
# print(t.sum(dim=1))

print(t.max())
print(t.max(dim=0))
print(t.max(dim=0)[0])
