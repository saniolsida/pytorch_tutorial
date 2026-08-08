import torch
import numpy as np

x_train = torch.FloatTensor([[1], [2], [3]])
y_train = torch.FloatTensor([[1], [2], [3]])

W = torch.zeros(1, requires_grad=True)
# lr = 0.1

nb_epochs = 10
# for epochs in range(1, nb_epochs + 1):
#     hypothesis = W * x_train

#     cost = torch.mean((hypothesis - y_train) ** 2)
#     gradient = torch.sum((W * x_train - y_train) * x_train)

#     print(f"Epoch {epochs}/{nb_epochs} W: {W.item():.3f}, Cost: {cost.item():.6f}")

#     W -= lr * gradient

optimizer = torch.optim.SGD([W], lr=0.15)
for epochs in range(nb_epochs + 1):
    hypothesis = W * x_train

    cost = torch.mean((hypothesis - y_train) ** 2)
    print(f"Epoch {epochs}/{nb_epochs} W: {W.item():.3f}, Cost: {cost.item():.6f}")

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()
