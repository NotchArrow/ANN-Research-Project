import os
import random

import torch
from torch import nn

import ModelData
from ModelData import Datasets, DataType


# force gpu usage for training and testing models
device = torch.device("cuda")
print(f"Using {device} device")

# force deterministic gpu behavior
os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True)

# generate seed
seed = random.randint(1, 1000000)
print(f"Using seed: {seed}")

# seed initial weights and shuffling
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# generate dataloaders
train_dataloader = ModelData.getLoader(dataset=Datasets.MNIST, batch_size=64, data_type=DataType.TRAINING, seed=seed)
test_dataloader = ModelData.getLoader(dataset=Datasets.MNIST, batch_size=64, data_type=DataType.TESTING, seed=seed)
for X, y in test_dataloader:
    print(f"Shape of X [N, C, H, W]: {X.shape}")
    print(f"Shape of y: {y.shape} {y.dtype}")
    break

# define model
class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()

        # set model architecture
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(28*28, 512),
            nn.ReLU(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# output model info and load to gpu
model = NeuralNetwork().to(device)
print(model)

# define training loop
def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# define testing algorithm for verification using testing data
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")

# default optimizer settings, will be changed during experimentation
optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

# run model training/testing for 5 epochs
epochs = 5
loss_fn = nn.CrossEntropyLoss()
for t in range(epochs):
    print(f"Epoch {t+1}\n-------------------------------")
    train(train_dataloader, model, loss_fn, optimizer)
    test(test_dataloader, model, loss_fn)
print("Done!")