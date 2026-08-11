# problem of sigmoid: gradient of side part is not valuable
# ReLU: f(x) = max(0,x), 음수 영역에서 gradient가 사라지는 문제가 있지만 잘 동작한다?
# torch.nn.relu(x), leaky_relu
# torch.optim:

# Weight initialization
# not 0
# Restricted Boltzmann Machine: 동일 집합안에서는 서로 연결 되어있지 않지만, 서로 다른 집합의 원소와 연결됨
# Xavier, He 입력 Neuron 수와 출력 neuron 수를 적절히 고려해서 적당한 크기로 랜덤 초기화

import torch

linear1 = torch.nn.Linear(784, 256, bias=True)
linear2 = torch.nn.Linear(256, 256, bias=True)
linear3 = torch.nn.Linear(256, 10, bias=True)
relu = torch.nn.ReLU()

torch.nn.init.xavier_uniform_(linear1.weight)
torch.nn.init.xavier_uniform_(linear2.weight)
torch.nn.init.xavier_uniform_(linear3.weight)
