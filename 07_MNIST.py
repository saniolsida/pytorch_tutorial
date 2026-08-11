import torch
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# MNIST 데이터
mnist_train = dsets.MNIST(
    root="MNIST_DATA/", train=True, transform=transforms.ToTensor(), download=True
)

mnist_test = dsets.MNIST(
    root="MNIST_DATA/", train=False, transform=transforms.ToTensor(), download=True
)

# 모델
linear = torch.nn.Linear(784, 10, bias=True).to(device)

training_epochs = 15
batch_size = 100

data_loader = torch.utils.data.DataLoader(
    dataset=mnist_train, batch_size=batch_size, shuffle=True, drop_last=True
)

criterion = torch.nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.SGD(linear.parameters(), lr=0.1)

# =========================
# Training
# =========================

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = len(data_loader)

    for X, Y in data_loader:
        X = X.view(-1, 28 * 28).to(device)
        Y = Y.to(device)

        optimizer.zero_grad()

        hypothesis = linear(X)

        cost = criterion(hypothesis, Y)

        cost.backward()

        optimizer.step()

        avg_cost += cost.item() / total_batch

    print(f"Epoch: {epoch + 1:04d}, Cost = {avg_cost:.6f}")


# =========================
# Test Accuracy
# =========================

linear.eval()

with torch.no_grad():
    X_test = mnist_test.data.view(-1, 28 * 28).float().to(device)
    X_test = X_test / 255.0

    Y_test = mnist_test.targets.to(device)

    prediction = linear(X_test)

    predicted_labels = torch.argmax(prediction, dim=1)

    accuracy = (predicted_labels == Y_test).float().mean()

    print(f"Accuracy: {accuracy.item() * 100:.2f}%")


# =========================
# Matplotlib 결과 확인
# =========================

plt.figure(figsize=(12, 5))

for i in range(10):
    image = mnist_test.data[i]
    label = mnist_test.targets[i].item()
    prediction = predicted_labels[i].item()

    plt.subplot(2, 5, i + 1)

    plt.imshow(image, cmap="gray")

    plt.title(f"Pred: {prediction}\nTrue: {label}")

    plt.axis("off")

plt.tight_layout()
plt.show()
