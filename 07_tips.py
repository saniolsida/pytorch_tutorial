import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

# |x_train| = (m,3)
# |y_train| = (m)
x_train = torch.FloatTensor(
    [
        [1, 2, 1],
        [1, 3, 2],
        [1, 3, 4],
        [1, 5, 5],
        [1, 7, 5],
        [1, 2, 5],
        [1, 6, 6],
        [1, 7, 7],
    ]
)

y_train = torch.LongTensor([2, 2, 2, 1, 1, 1, 0, 0])

# |x_test| = (m',3)
x_test = torch.FloatTensor([[2, 1, 1], [3, 1, 2], [3, 3, 4]])
y_test = torch.LongTensor([2, 2, 2])


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, x):
        return self.linear(x)


model = SoftmaxClassifierModel()
optimizer = optim.SGD(model.parameters(), lr=0.1)


def train(model, optimizer, x_train, y_train):
    nb_epochs = 20
    for epochs in range(nb_epochs):
        prediction = model(x_train)

        cost = F.cross_entropy(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print(f"Epoch {epochs:4d} cost: {cost.item():.6f}")


def test(model, optimizer, x_test, y_test):
    prediction = model(x_test)
    predicted_classes = prediction.max(1)[1]
    correct_count = (predicted_classes == y_test).sum().item()
    cost = F.cross_entropy(prediction, y_test)

    print(f"Accuracy: {correct_count / len(y_test) * 100} Cost: {cost.item():.6f}")


train(model, optimizer, x_train, y_train)
test(model, optimizer, x_test, y_test)
## overfitting의 사례.
