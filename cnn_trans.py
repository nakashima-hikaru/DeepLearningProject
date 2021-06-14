import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import tqdm

from torchvision.datasets import FashionMNIST, ImageFolder
from torchvision import transforms, models

my_path = '/Users/nakashimahikaru/PycharmProjects/DeepLearningProject/taco_and_burrito'
train_imgs = ImageFolder("{}/train/".format(my_path),
                         transform=transforms.Compose([transforms.RandomCrop(224), transforms.ToTensor()]))
test_imgs = ImageFolder("{}/test/".format(my_path),
                         transform=transforms.Compose([transforms.CenterCrop(224), transforms.ToTensor()]))

train_loader = DataLoader(train_imgs, batch_size=32, shuffle=True)
test_loader = DataLoader(test_imgs, batch_size=32, shuffle=False)

net = models.resnet18(pretrained=True)
for p in net.parameters():
    p.requires_grad=False

fc_input_dim = net.fc.in_features
net.fc = nn.Linear(fc_input_dim, out_features=2)

def eval_net(net, data_loader, device="cpu"):
    net.eval()
    ys = []
    ypreds = []
    for x, y in data_loader:
        x = x.to(device)
        y = y.to(device)
        with torch.no_grad():
            _, y_pred = net(x).max(1)
        ys.append(y)
        ypreds.append(y_pred)
        ys = torch.cat(ys)
        ypreds = torch.cat(ypreds)
        acc = (ys == ypreds).float().sum() / len(ys)
        return acc.item()

def train_net(net, train_loader, test_loader, only_fc=True, optimizer_cls=optim.Adam, loss_fn=nn.CrossEntropyLoss(), n_iter=3, device="cpu"):
    train_losses = []
    train_acc = []
    val_acc = []
    if only_fc:
        optimizer_cls(net.fc.parameters())
    else:
        optimizer_cls(net.parameters())
    optimizer = optimizer_cls(net.parameters())
    for epoch in range(n_iter):
        running_loss = 0.0
        net.train()
        n = 0
        n_acc = 0
        for i, (xx, yy) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            xx=xx.to(device)
            yy = yy.to(device)
            h = net(xx)
            loss = loss_fn(h, yy)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            n += len(xx)
            _, y_pred = h.max(axis=1)
            n_acc += (yy == y_pred).float().sum().item()
        train_losses.append(running_loss / i)

        train_acc.append(n_acc / n)
        val_acc.append(eval_net(net, test_loader, device))
        print(epoch, train_losses[-1], train_acc[-1], val_acc[-1], flush=True)

train_net(net, train_loader, test_loader, n_iter=3, device="cpu")


