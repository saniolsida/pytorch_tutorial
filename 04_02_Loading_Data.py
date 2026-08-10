# Multivariate Linear Regression: 다양한 변수로 하나의 결과를 도출

# 엄청난 양의 데이터를 한번에 학습시킬 수 없으니 일부분만 갖고 학습하면 어떨까?
# => Minibatch Gradient Descent
# 업데이트가 빠르다. 전체 데이터를 쓰지 않아서 잘못된 방향으로 학습할 수 있다.

import torch
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.nn as nn


class MultivariateLinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 1)

    def forward(self, x):
        return self.linear(x)


class CustomDataset(Dataset):
    def __init__(self):
        self.x_data = [
            [73, 80, 75],
            [93, 88, 93],
            [89, 91, 90],
            [96, 98, 100],
            [73, 66, 70],
        ]
        self.y_data = [[152], [185], [180], [196], [142]]

    def __len__(self):
        return len(self.x_data)

    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])
        y = torch.FloatTensor(self.y_data[idx])

        return x, y


dataset = CustomDataset()

dataloader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=True,  # 데이터가 학습되는 순서를 섞어 정답을 외우는 것을 방지한다.
)

model = MultivariateLinearRegressionModel()
optimizer = torch.optim.SGD(model.parameters(), lr=1e-5)

nb_epochs = 20
for epochs in range(nb_epochs + 1):
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples

        prediction = model(x_train)

        cost = F.mse_loss(prediction, y_train)

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        print(
            f"Epoch {epochs:.4f} Batch {batch_idx + 1}/{len(dataloader)} Cost: {cost.item():.6f}"
        )
