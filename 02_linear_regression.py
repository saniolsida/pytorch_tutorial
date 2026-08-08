import torch

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[2], [4], [6]])

#  y = Wx + b // W = weight, b = bias
W = torch.zeros(1, requires_grad=True)
b = torch.zeros(1, requires_grad=True)

# compute loss
# Mean Squared Error(MSE) loss 계산하는 함수

optimizer = torch.optim.SGD([W, b], lr=0.01)

nb_epochs = 1000
for epoch in range(1, nb_epochs + 1):
    hypothesis = x_train * W + b
    cost = torch.mean((hypothesis - y_train) ** 2)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

# hypothesis = xW + b
# 얼마나 틀렸는지 계산 -> 이전 미분값 삭제 -> 어떻게 수정해야하는지 미분 -> 실제 W,b 수정

print("W = ", W.item())
print("b = ", b.item())
print("cost = ", cost.item())
