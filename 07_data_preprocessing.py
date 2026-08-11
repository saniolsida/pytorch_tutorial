import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


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


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


mu = x_train.mean(dim=0)
sigma = x_train.std(dim=0)
norm_x_train = (x_train - mu) / sigma
# print(norm_x_train)

model = MultivariateLinearRegressionModel()
optimizer = optim.SGD(model.parameters(), lr=1e-1)


def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epochs in range(nb_epochs):
        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print(f"Epoch {epochs:4d} cost: {cost.item():.6f}")


train(model, optimizer, norm_x_train, y_train)
