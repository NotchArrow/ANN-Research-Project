import os
import time
import warnings
from datetime import datetime

import pandas as pd
import torch
from numpy.exceptions import VisibleDeprecationWarning
from torch import nn
from torchmetrics.classification import MulticlassF1Score

import TrialManager


# define model
class NeuralNetwork(nn.Module):
    def __init__(self, architecture):
        super().__init__()
        self.flatten = nn.Flatten()

        # set model architecture
        self.linear_relu_stack = architecture

    def forward(self, x):
        x = self.flatten(x)
        logits = self.linear_relu_stack(x)
        return logits

# define training loop
def train(dataloader, model, loss_fn, optimizer):
    total_correct = 0
    total_samples = 0
    running_loss = 0.0
    f1_metric = MulticlassF1Score(num_classes=10, average='macro').to(device, non_blocking=True)

    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)

        # compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # record metrics
        predicted = torch.max(pred, 1)[1]
        total_correct += (predicted == y).sum().item()
        total_samples += y.size(0)
        running_loss += loss.item() * y.size(0)
        f1_metric.update(pred, y)

        # backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_accuracy = total_correct / total_samples
    epoch_f1 = f1_metric.compute().item()
    return epoch_loss, epoch_accuracy, epoch_f1

# define testing algorithm for verification using testing data
def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()

    f1_metric = MulticlassF1Score(num_classes=10, average='macro').to(device, non_blocking=True)
    test_loss, correct = 0, 0
    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device, non_blocking=True), y.to(device, non_blocking=True)
            pred = model(X)
            test_loss += loss_fn(pred, y).item()
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()
            f1_metric.update(pred, y)
    test_loss /= num_batches
    correct /= size
    f1 = f1_metric.compute().item()

    return test_loss, correct, f1

if __name__ == "__main__":
    # force gpu usage for training and testing models
    device = torch.device("cuda")
    print(f"Using {device} device")
    print("Starting...\n")

    # force deterministic gpu behavior
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)

    # supress numpy deprecation warning for cifar10 (caused by torchvision)
    warnings.filterwarnings("ignore", category=VisibleDeprecationWarning)

    # get next trial to complete and start loop
    trial_data = TrialManager.getTrial()
    while trial_data is not None:
        train_dataloader, test_dataloader, lr, optimizer_type, architecture, seed, load_time, start_index = trial_data

        # load model to gpu
        model = NeuralNetwork(architecture=architecture).to(device)

        # setup model optimizer and loss function
        optimizer = optimizer_type(model.parameters(), lr=lr)
        loss_fn = nn.CrossEntropyLoss()

        # setup logging
        runtimes = []
        training_losses = []
        training_accuracies = []
        training_f1 = []
        testing_losses = []
        testing_accuracies = []
        testing_f1 = []

        # run model training/testing for 50 epochs
        epochs = 50
        for t in range(epochs):
            epoch_start_time = time.time()

            train_metrics = train(train_dataloader, model, loss_fn, optimizer)
            training_losses.append(train_metrics[0])
            training_accuracies.append(train_metrics[1])
            training_f1.append(train_metrics[2])

            test_metrics = test(test_dataloader, model, loss_fn)
            testing_losses.append(test_metrics[0])
            testing_accuracies.append(test_metrics[1])
            testing_f1.append(test_metrics[2])

            runtimes.append(time.time() - epoch_start_time)

        # log data
        dataFrame = pd.read_csv("trials.csv", dtype=float)

        for i, index in enumerate(range(start_index, start_index + epochs)):
            row = dataFrame.iloc[index]
            row["Seed"] = seed
            row["Data Load Time (s)"] = load_time
            row["Runtime (s)"] = runtimes[i]
            row["Loss (Training)"] = training_losses[i]
            row["Accuracy (Training)"] = training_accuracies[i]
            row["F1 Score (Training)"] = training_f1[i]
            row["Loss (Testing)"] = testing_losses[i]
            row["Accuracy (Testing)"] = testing_accuracies[i]
            row["F1 Score (Testing)"] = testing_f1[i]

        dataFrame.to_csv("trials.csv", index=False)
        print(f"Finished Trial At: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # get new trial info
        trial_data = TrialManager.getTrial()