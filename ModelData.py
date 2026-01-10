from enum import Enum

import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor


class Datasets(Enum):
    MNIST = datasets.MNIST
    FASHION_MNIST = datasets.FashionMNIST
    CIFAR10 = datasets.CIFAR10

class DataType(Enum):
    TRAINING = True
    TESTING = False

def getLoader(dataset, batch_size, data_type, seed):
    dataset = dataset.value
    data_type = data_type.value

    return DataLoader(dataset(root="data",
                              train=data_type,
                              download=True,
                              transform=ToTensor()),
                      batch_size=batch_size,
                      shuffle=True,
                      generator=torch.Generator().manual_seed(seed))