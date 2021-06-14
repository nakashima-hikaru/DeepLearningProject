import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import tqdm
from torchvision.datasets import FashionMNIST, ImageFolder, CIFAR10
from torchvision import transforms, models


# download the dataset
my_path = '/Users/nakashimahikaru/PycharmProjects/DeepLearningProject/taco_and_burrito'
cifar10_train = CIFAR10(root='{}/cifar-10'.format(my_path), train=True, download=False, transform=transforms.ToTensor())
cifar10_test = CIFAR10(root='{}/cifar-10'.format(my_path), train=False, download=False, transform=transforms.ToTensor())

batch_size = 128  # ToDo: optimize the batch size
train_loader = torch.utils.data.DataLoader(cifar10_train,  batch_size=batch_size,  shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar10_test,  batch_size=batch_size,  shuffle=False)

# check DataLoader
tmp = train_loader.__iter__()
x1, y1 = tmp.next()
print(x1, y1)