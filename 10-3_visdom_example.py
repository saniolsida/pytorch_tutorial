import torch
import torch.nn as nn
import torchvision.transforms as transforms

import torchvision
import torchvision.datasets as dsets

import visdom

vis = visdom.Visdom()


MNIST = dsets.MNIST(
    root="./MNIST_data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
cifar10 = dsets.CIFAR10(
    root="./cifar10",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)


def loss_tracker(loss_plot, loss_value, num):
    """num, loss_value, are Tensor"""
    vis.line(X=num, Y=loss_value, win=loss_plot, update="append")


if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

torch.manual_seed(777)
if device == "cuda":
    torch.cuda.manual_seed_all(777)

print(device)

learning_rate = 0.001
training_epochs = 15
batch_size = 100

mnist_train = dsets.MNIST(
    root="MNIST_data/", train=True, transform=transforms.ToTensor(), download=True
)

mnist_test = dsets.MNIST(
    root="MNIST_data/", train=False, transform=transforms.ToTensor(), download=True
)

data_loader = torch.utils.data.DataLoader(
    dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
)


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.fc1 = nn.Linear(3 * 3 * 128, 625)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(625, 10, bias=True)
        torch.nn.init.xavier_uniform_(self.fc1.weight)
        torch.nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)

        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)

        return out


model = CNN().to(device)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

loss_plt = vis.line(
    Y=torch.Tensor(1).zero_(),
    opts=dict(title="loss_tracker", legend=["loss"], showlegend=True),
)

total_batch = len(data_loader)

for epoch in range(training_epochs):
    avg_cost = 0

    for X, Y in data_loader:
        X = X.to(device)
        Y = Y.to(device)

        optimizer.zero_grad()
        hypothesis = model(X)

        cost = criterion(hypothesis, Y)
        cost.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Cost: {avg_cost}")
    loss_tracker(loss_plt, torch.Tensor([avg_cost]), torch.Tensor([epoch]))
print("Learning Finished!")

with torch.no_grad():
    X_test = mnist_test.test_data.view(len(mnist_test), 1, 28, 28).float().to(device)
    Y_test = mnist_test.test_labels.to(device)

    prediction = model(X_test)
    correct_prediction = torch.argmax(prediction, 1) == Y_test
    accuracy = correct_prediction.float().mean()
    print(f"Accuracy: {accuracy.item()}")
