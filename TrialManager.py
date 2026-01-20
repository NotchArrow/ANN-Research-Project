import random
import time
from datetime import datetime

import pandas as pd
import torch
from torch import optim, nn
from torch.utils.data import TensorDataset, DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

if __name__ == "__main__":
    print("Generating trials spreadsheet...")
    data = {}

    headers = ["Batch Size", "Dataset ID", "Learning Rate", "Optimizer ID", "Architecture ID", "Trial Number", "Epoch",
               "Seed", "Data Load Time (s)", "Runtime (s)",
               "Loss (Training)", "Accuracy (Training)", "F1 Score (Training)",
               "Loss (Testing)", "Accuracy (Testing)", "F1 Score (Testing)"]
    i = 1
    for batchSize in [64, 128, 256]:
        for dataset in [0, 1, 2]:  # MNIST, FASHION, CIFAR
            for learningRate in [0.01, 0.001, 0.0001]:
                for optimizer in [0, 1]:  # SGD, ADAM
                    for architecture in range(20):
                        for trial in range(1, 4):
                            for epoch in range(1, 51):
                                data[i] = [batchSize, dataset, learningRate, optimizer, architecture, trial, epoch,
                                           0, 0, 0,
                                           0, 0, 0,
                                           0, 0, 0]
                                i += 1

    dataFrame = pd.DataFrame(
        data
    )

    print(dataFrame.T)
    dataFrame.T.to_csv("new_trials.csv", header=headers, index=False)
    print("Please rename your spreadsheet to 'trials.csv' to avoid accidental overwrite!")


def getTrial():
    dataFrame = pd.read_csv("trials.csv", dtype=float)
    rowCount = dataFrame.shape[0]

    for index, row in dataFrame.iterrows():
        if row["Epoch"] == 1 and row["Seed"] == 0:
            start_index = index
            batchSize = int(row["Batch Size"])
            datasetID = row["Dataset ID"]
            learningRate = row["Learning Rate"]
            optimizerID = row["Optimizer ID"]
            architectureID = row["Architecture ID"]

            match datasetID:
                case 0:
                    train_dataset = datasets.MNIST(root="data", train=True, download=True, transform=ToTensor())
                    test_dataset = datasets.MNIST(root="data", train=False, download=True, transform=ToTensor())
                    input_size = 784
                case 1:
                    train_dataset = datasets.FashionMNIST(root="data", train=True, download=True, transform=ToTensor())
                    test_dataset = datasets.FashionMNIST(root="data", train=False, download=True, transform=ToTensor())
                    input_size = 784
                case 2:
                    train_dataset = datasets.CIFAR10(root="data", train=True, download=True, transform=ToTensor())
                    test_dataset = datasets.CIFAR10(root="data", train=False, download=True, transform=ToTensor())
                    input_size = 3072

            seed = random.randint(1, 1000000)
            print("Trial Info:")
            print(row[:6])
            print(f"Using seed: {seed}")
            print(f"Total trial completion: {(100 * index / rowCount):>0.2f}%")

            # seed initial weights and shuffling
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)

            start_load_time = time.time()
            loaded_train = DataLoader(
                TensorDataset(
                    torch.stack([img for img, _ in train_dataset]),
                    torch.tensor([label for _, label in train_dataset])
                ),
                batch_size=batchSize,
                shuffle=True,
                generator=torch.Generator().manual_seed(seed),
            )
            loaded_test = DataLoader(
                TensorDataset(
                    torch.stack([img for img, _ in test_dataset]),
                    torch.tensor([label for _, label in test_dataset])
                ),
                batch_size=batchSize,
                shuffle=False,
                generator=torch.Generator().manual_seed(seed),
            )
            load_time = time.time() - start_load_time

            match optimizerID:
                case 0:
                    optimizer = optim.SGD
                case 1:
                    optimizer = optim.Adam

            match architectureID:
                case 0:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 10)
                    )
                case 1:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 20),
                        nn.ReLU(),
                        nn.Linear(20, 10)
                    )
                case 2:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                    )
                case 3:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, 10)
                    )
                case 4:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 10)
                    )
                case 5:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 10)
                    )
                case 6:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                    )
                case 7:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 256),
                        nn.ReLU(),
                        nn.Linear(256, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10)
                    )
                case 8:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10)
                    )
                case 9:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 10)
                    )
                case 10:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 256),
                        nn.ReLU(),
                        nn.Linear(256, 10)
                    )
                case 11:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, 128),
                        nn.ReLU(),
                        nn.Linear(128, 32),
                        nn.ReLU(),
                        nn.Linear(32, 10)
                    )
                case 12:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10)
                    )
                case 13:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                    )
                case 14:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 10)
                    )
                case 15:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10)
                    )
                case 16:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 32),
                        nn.ReLU(),
                        nn.Linear(32, 10)
                    )
                case 17:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 10)
                    )
                case 18:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 32),
                        nn.ReLU(),
                        nn.Linear(32, 10)
                    )
                case 19:
                    architecture = nn.Sequential(
                        nn.Linear(input_size, 1024),
                        nn.ReLU(),
                        nn.Linear(1024, 512),
                        nn.ReLU(),
                        nn.Linear(512, 256),
                        nn.ReLU(),
                        nn.Linear(256, 128),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU(),
                        nn.Linear(64, 10)
                    )

            print(f"Started Trial At: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} | Load Time Elapsed: {round(load_time, 2)}s")
            return loaded_train, loaded_test, learningRate, optimizer, architecture, seed, load_time, start_index
    return None