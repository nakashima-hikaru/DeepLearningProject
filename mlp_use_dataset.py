import torch
from torch import nn, optim
from sklearn.datasets import load_digits
from torch.utils.data import (Dataset, DataLoader, TensorDataset)

net = nn.Sequential(nn.Linear(64, 32),
                    nn.ReLU(),
                    nn.Linear(32, 16),
                    nn.ReLU(),
                    nn.Linear(16, 10)
                    )

digits = load_digits()
X = digits.data
y = digits.target
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.int64)
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(net.parameters())

ds = TensorDataset(X, y)
loader = DataLoader(ds, batch_size=64, shuffle=True)

losses = []
for epoc in range(10):
    runnning_loss = 0.0
    for xx, yy in loader:
        y_pred = net(xx)
        loss = loss_fn(y_pred, yy)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        runnning_loss += loss.item()
    losses.append(runnning_loss)

from matplotlib import  pyplot as plt
plt.plot(losses)
plt.show()


