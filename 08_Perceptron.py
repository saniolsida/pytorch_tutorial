# Perceptron: 인공 신경망의 한 종류
# Neuron: 인간의 뇌에서 신호를 전달하는 역할 수행
# Linear classifier를 위해 개발함. AND, OR를 위해 개발함
# Linear한 방식으로는 XOR 구분을 할 수 없다.

import torch

torch.manual_seed(3)
device = "cuda" if torch.cuda.is_available() else "cpu"

X = torch.FloatTensor([[0, 0], [0, 1], [1, 0], [1, 1]]).to(device)
Y = torch.FloatTensor([[0], [1], [1], [0]]).to(device)

linear1 = torch.nn.Linear(2, 2, bias=True)
linear2 = torch.nn.Linear(2, 1, bias=True)

sigmoid = torch.nn.Sigmoid()
model = torch.nn.Sequential(linear1, sigmoid, linear2, sigmoid).to(device)

criterion = torch.nn.BCELoss().to(device)
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
for step in range(10001):
    optimizer.zero_grad()
    hypothesis = model(X)

    cost = criterion(hypothesis, Y)
    cost.backward()
    optimizer.step()
    if step % 100 == 0:
        print(step, cost.item())


with torch.no_grad():
    hypothesis = model(X)
    print("Prediction: ", hypothesis)
    print("Binary: ", (hypothesis > 0.5).float())
    print("Answer: ", Y)
