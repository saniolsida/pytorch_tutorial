import torch

import torch.nn as nn
import torch.nn.functional as F

# Multivariate Linear Regression
# H = w1x1 + w2x2 .. wnxn + b

# 만약 x의 길이가 1000이라면?
# matmul(x) 를 사용한다.
# cost는 기존과 동일한 공식

# x_train = torch.FloatTensor(
#     [
#         [73, 80, 75],
#         [93, 88, 93],
#         [89, 91, 90],
#         [96, 98, 100],
#         [73, 66, 70],
#     ]
# )
# y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

# W = torch.zeros((3, 1), requires_grad=True)
# b = torch.zeros(1, requires_grad=True)

# optimizer = torch.optim.SGD([W, b], lr=1e-5)

# nb_epochs = 20
# for epochs in range(nb_epochs + 1):
#     hypothesis = x_train.matmul(W) + b

#     cost = torch.mean((hypothesis - y_train) ** 2)

#     optimizer.zero_grad()
#     cost.backward()
#     optimizer.step()

#     print(
#         f"Epoch {epochs:.4f}/{nb_epochs} hypothesis: {hypothesis.squeeze()} Cost: {cost.item():.6f}"
#     )


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


x_train = torch.FloatTensor(
    [
        [73, 80, 75],
        [93, 88, 93],
        [89, 91, 90],
        [96, 98, 100],
        [73, 66, 70],
    ]
)
y_train = torch.FloatTensor([[152], [185], [180], [196], [142]])

model = MultivariateLinearRegressionModel()

optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epochs in range(nb_epochs + 1):
    Hypothesis = model(x_train)

    cost = F.mse_loss(Hypothesis, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    print(
        f"Epoch {epochs:.4f}/{nb_epochs} Hypothesis: {Hypothesis.squeeze()}, Cost: {cost.item():.6f}"
    )
