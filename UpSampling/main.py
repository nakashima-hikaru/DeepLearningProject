import torch
from torch import nn, optim
from torch.utils.data import (Dataset, DataLoader, TensorDataset)
import tqdm
from torchvision.datasets import FashionMNIST, ImageFolder, CIFAR10
from torchvision import transforms, models
from util import check_data_loader_contents


# download the dataset
my_path = '/Users/nakashimahikaru/PycharmProjects/DeepLearningProject/dataset/'
cifar10_train = CIFAR10(root='{}/cifar-10'.format(my_path), train=True, transform=transforms.ToTensor())
cifar10_test = CIFAR10(root='{}/cifar-10'.format(my_path), train=False, transform=transforms.ToTensor())

batch_size = 128  # ToDo: optimize the batch size
train_loader = torch.utils.data.DataLoader(cifar10_train,  batch_size=batch_size,  shuffle=True)
test_loader = torch.utils.data.DataLoader(cifar10_test,  batch_size=batch_size,  shuffle=False)
check_data_loader_contents(train_loader)

test_input = torch.ones(128, 3, 32, 32)

conv_net = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=32, kernel_size=(5, )),  # [128, 32, 28, 28]
                         nn.MaxPool2d(kernel_size=2),  # [128, 32, 14, 14]
                         nn.ReLU(),  # [128, 32, 14, 14]
                         # nn.Conv2d(in_channels=3, out_channels=10, kernel_size=(5, )),
                         )

conv_output_size = conv_net(test_input).size()[-1]
print(conv_net(test_input).size())
