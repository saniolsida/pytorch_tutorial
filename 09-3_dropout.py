# overfitting: 과적합
# train에 최적화되어서 Test set에서 오분류가 일어남
# Dropout: 일부 뉴런을 랜덤하게 끄는 방법이다. 특정 뉴런 몇 개에 모델이 너무 의존하는걸 막는다.

import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

linear1 = torch.nn.Linear(784, 512, bias=True)
linear2 = torch.nn.Linear(512, 512, bias=True)
linear3 = torch.nn.Linear(512, 512, bias=True)
linear4 = torch.nn.Linear(512, 512, bias=True)
linear5 = torch.nn.Linear(512, 10, bias=True)
relu = torch.nn.ReLU()
# dropout = torch.nn.Dropout(p=drop_prop)

# model = torch.nn.Sequential(
#     linear1,
#     relu,
#     dropout,
#     linear2,
#     relu,
#     dropout,
#     linear3,
#     relu,
#     dropout,
#     linear4,
#     relu,
#     dropout,
#     linear5,
# ).to(device)

# gradient vanishing
# covariate shift: train, test의 분포 차이가 있다. 그 차이가 문제를 발생시킨다.

# Batch Normalization: 신경망 내부에서 각 층의 출력값을 정규화
