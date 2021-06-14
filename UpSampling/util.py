import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import tqdm
from torchvision.datasets import FashionMNIST, ImageFolder, CIFAR10
from torchvision import transforms, models

def check_data_loader_contents(data_loader):
    tmp = data_loader.__iter__()
    x1, y1 = tmp.next()
    print('data size:{}'.format(len(data_loader)))
    print('shape of x:{}'.format(x1.shape))
    print('type of x:{}'.format(x1.dtype))
    print('shape of y:{}'.format(y1.shape))
    print('type of y:{}'.format(y1.dtype))
