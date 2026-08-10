# softmax:
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)  # 난수 생성을 고정한다.

z = torch.FloatTensor([1, 2, 3])

hypothesis = F.softmax(z, dim=0)
# print(hypothesis)

# print(hypothesis.sum())

z = torch.rand(3, 5, requires_grad=True)
hypothesis = F.softmax(z, dim=1)
# print(hypothesis)

y = torch.randint(5, (3,)).long()
# print(y)

# one hot vector: 정답은 1, 나머지는 0으로 표현

# y_one_hot = torch.zeros_like(hypothesis)
# print(y_one_hot.scatter_(1, y.unsqueeze(1), 1))

# cost = (y_one_hot * -torch.log(hypothesis)).sum(dim=1).mean()
# print(cost)

# cost = F.nll_loss(F.log_softmax(z, dim=1), y)
# print(cost)


class SoftmaxClassifierModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(self, x):
        return self.linear(x)


x_train = [
    [1, 2, 1, 1],
    [2, 1, 3, 2],
    [3, 1, 3, 4],
    [4, 1, 5, 5],
    [1, 7, 5, 5],
    [1, 2, 5, 6],
    [1, 6, 6, 6],
    [1, 7, 7, 7],
]
y_train = [2, 2, 2, 1, 1, 1, 0, 0]
x_train = torch.FloatTensor(x_train)
y_train = torch.LongTensor(y_train)

W = torch.zeros((4, 3), requires_grad=True)
b = torch.zeros(1, requires_grad=True)

model = SoftmaxClassifierModel()
# optimizer = optim.SGD([W, b], lr=0.1)
optimizer = optim.SGD(model.parameters(), lr=0.1)

nb_epochs = 1000
for epochs in range(nb_epochs + 1):
    prediction = model(x_train)
    # z = x_train.matmul(W) + b
    # cost = F.cross_entropy(z, y_train)

    cost = F.cross_entropy(prediction, y_train)

    optimizer.zero_grad()
    cost.backward()
    optimizer.step()

    if epochs % 100 == 0:
        print(f"Cost: {cost.item():.6f}")
