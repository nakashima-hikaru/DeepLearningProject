import torch
from torch import nn, optim
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data[:100]
y = iris.target[:100]
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32)
net = nn.Linear(in_features=4, out_features=1)
loss_fn = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(net.parameters(), lr=0.25)

losses = []
for epoc in range(100):
    optimizer.zero_grad()
    y_pred = net(X)
    loss = loss_fn(y_pred.view_as(y), y)
    loss.backward()
    optimizer.step()
    losses.append(loss.item())

from matplotlib import pyplot as plt
plt.plot(losses)
plt.show()