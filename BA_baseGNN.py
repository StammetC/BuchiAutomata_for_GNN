from abc import ABC

import networkx as nx
import torch
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import torch_geometric.data.dataset
from torch_geometric.utils.convert import to_networkx
from torch_geometric.data import Data, InMemoryDataset
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader
from torch_geometric.nn import GCNConv, GENConv, RGCNConv, GATConv
from torch.nn import Linear
import torch.nn.functional as F
from typing import Tuple
from torch_geometric.nn import global_mean_pool, global_max_pool
from datetime import datetime
from collections import Counter
import sys

dataset_folder = "datasets/paperdatasets/newedgeattr"
if not os.path.exists("traininggraphs"):
    os.makedirs("traininggraphs")
if not os.path.exists("results"):
    os.makedirs("results")


class Dataset(InMemoryDataset, ABC):
    def __init__(self, data_list, id):
        self.id = id
        super(Dataset, self).__init__('./data/Dataset')
        self.data, self.slices = self.collate(data_list)


class BA_GCN(torch.nn.Module):
    def __init__(self, hidden_channels, num_node_features, num_classes):
        super(BA_GCN, self).__init__()
        self.hidden_channels = hidden_channels
        # torch.manual_seed(527)
        self.conv1 = RGCNConv(num_node_features, hidden_channels, 2)
        self.conv2 = RGCNConv(hidden_channels, hidden_channels, 2)
        self.conv3 = RGCNConv(hidden_channels, hidden_channels, 2)
        self.lin = Linear(hidden_channels, num_classes)

    def forward(self, x, edge_index, edge_attr, batch):
        # 1. Obtain node embeddings
        x = self.conv1(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv2(x, edge_index, edge_attr)
        x = x.relu()
        x = self.conv3(x, edge_index, edge_attr)
        x = x.relu()
        # 2. Readout layer
        x = global_mean_pool(x, batch)  # [batch_size, hidden_channels]
        # 3. Apply a final classifier
        x = self.lin(x)
        return x


def load_trainset(trainset_filename: str = None) -> torch_geometric.data.dataset.Dataset:
    items = os.listdir(dataset_folder)
    fileList = [name for name in items]
    if trainset_filename in fileList:
        print(f"Training dataset {trainset_filename} was loaded successfully!")
        train_dataset = Dataset(torch.load(dataset_folder + "/" + trainset_filename), trainset_filename).shuffle()
    else:
        for cnt, fileName in enumerate(fileList, 1):
            sys.stdout.write("[%d] %s\n\r" % (cnt, fileName))
        choice = int(input("Select training dataset[1-%s]: " % cnt)) - 1
        train_dataset = Dataset(torch.load(dataset_folder + "/" + fileList[choice]), fileList[choice]).shuffle()
    return train_dataset


def load_testset(testset_filename: str = None) -> torch_geometric.data.dataset.Dataset:
    items = os.listdir(dataset_folder)
    fileList = [name for name in items]
    if testset_filename in fileList:
        print(f"Testing dataset {testset_filename} was loaded successfully!")
        test_dataset = Dataset(torch.load(dataset_folder + "/" + testset_filename), testset_filename).shuffle()
    else:
        for cnt, fileName in enumerate(fileList, 1):
            sys.stdout.write("[%d] %s\n\r" % (cnt, fileName))

        choice = int(input("Select testing dataset[1-%s]: " % cnt)) - 1
        test_dataset = Dataset(torch.load(dataset_folder + "/" + fileList[choice]), fileList[choice]).shuffle()
    return test_dataset


def load_datasets(trainset_filename: str = None, testset_filename: str = None) -> \
        Tuple[torch_geometric.data.dataset.Dataset, torch_geometric.data.dataset.Dataset]:
    train_dataset = load_trainset(trainset_filename)
    test_dataset = load_testset(testset_filename)
    return train_dataset, test_dataset


def avg_no_of_nodes(dataset: torch_geometric.data.dataset.Dataset) -> float:
    total = 0
    for d in dataset:
        total += d.num_nodes
    return total / len(dataset)


def train(model, loader):
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = torch.nn.CrossEntropyLoss()
    model.train()
    for data in loader:  # Iterate in batches over the training dataset.
        feature_start = data.x
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
        loss = criterion(out, data.y)  # Compute the loss.
        loss.backward()  # Derive gradients.
        optimizer.step()  # Update parameters based on gradients.
        optimizer.zero_grad()  # Clear gradients.
        feature_end = data.x
        # print(f"startfeat = {feature_start}, endfeat = {feature_end}")


def test(model, loader):
    model.eval()
    count_correct = 0
    count_errors = 0
    count_err_empty = 0
    count_err_nonempty = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        c_e = 0
        c_ne = 0
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        errors = ((abs(pred - data.y) == 1).nonzero(as_tuple=True)[0])
        for i in errors:
            if data.y[i] == 0:
                c_e += 1
            if data.y[i] == 1:
                c_ne += 1
        count_correct += int((pred == data.y).sum())  # Check against ground-truth labels.
        count_errors += len(errors)
        count_err_nonempty += c_ne
        count_err_empty += c_e
    # print(f"Number of errors: {count_errors}/{len(loader.dataset)} --- {count_err_empty} empty / {count_err_nonempty} non-empty")
    return count_correct / len(
        loader.dataset), count_errors, count_err_empty, count_err_nonempty  # Derive ratio of correct predictions.


def get_erroneous_classified_data(model, loader):
    errs = []
    to_accs = []
    acc_cycles = []
    for data in loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)
        pred = out.argmax(dim=1)  # Use the class with highest probability.
        errors = ((abs(pred - data.y) == 1).nonzero(as_tuple=True)[0])
        for i in errors:
            d = data.get_example(i)
            errs.append(d)
            to_accs.append(d.cyclelen[0])
            acc_cycles.append(d.cyclelen[1])
    return errs, Counter(to_accs), Counter(acc_cycles)


def get_avg_loss_over_dataset(model, loader):
    criterion = torch.nn.CrossEntropyLoss()
    loss = []
    for data in loader:
        out = model(data.x, data.edge_index, data.edge_attr, data.batch)  # Perform a single forward pass.
        loss.append(criterion(out, data.y))  # Compute the loss.
    return sum(loss) / len(loss)


def train_model_and_save_stats(model: BA_GCN, epochs: int,
                               trainset: torch_geometric.data.dataset.Dataset,
                               testset: torch_geometric.data.dataset.Dataset,
                               batch_size: int, save_training_graphs: bool, logging: bool):
    """
    hello

    :param model: The NN model that is to be trained
    :param epochs: The number of epochs for training
    :param trainset: The training dataset
    :param testset:  The testing dataset
    :param batch_size: The batch size of the data loading procedure
    :param save_training_graphs: If True, saves accuracy and loss fct graph to ./traininggraphs
    :return: None: Trains the given NN model
    """
    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=True)
    if logging:
        log = open("./results/results.txt", "a")
        log.write("=======================================\n")
        log.write("Start training of the network:\n")
        log.write(f"epochs: {str(epochs)}, batchsize: {str(batch_size)}, hiddenchannels: {str(model.hidden_channels)}\n")
        log.write(f"Training dataset: {trainset.id}, Test dataset: {testset.id}\n")
    trainacceval = []
    testacceval = []
    losseval = []
    for epoch in range(0, epochs):
        train(model, train_loader)
        train_acc, _, _, _ = test(model, train_loader)
        trainacceval.append(train_acc)
        test_acc, n_err, n_e, n_ne = test(model, test_loader)
        testacceval.append(test_acc)
        lossavg = get_avg_loss_over_dataset(model, train_loader)
        losseval.append(float(lossavg))

        print(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, '
              f'Test Acc: {test_acc:.4f} (e: {n_e}/ n-e: {n_ne}) - Loss: {lossavg}')
        if logging: log.write(f'Epoch: {epoch:03d}, Train Acc: {train_acc:.4f}, '
                              f'Test Acc: {test_acc:.4f} (e: {n_e}/ n-e: {n_ne}) - Loss: {lossavg}\n')

    mistakes, acc_lens, acc_cycles = get_erroneous_classified_data(model, test_loader)

    if logging:
        log.write("========== Training complete ==========\n")
        log.write(f"Final accuracy on testset: {str(100*test(model, test_loader)[0])}%\n")
    if save_training_graphs:
        plt.title(
            f"Trainset: {trainset.id} \n Testset: {testset.id} \n "
            f"Accuracy evolution: Channels: {model.hidden_channels}, Batchsize: {BATCH_SIZE}, Epochs: {epochs}\n ")
        plt.xlabel("No of epoch")
        plt.ylabel("Accuracy in %")
        plt.plot(trainacceval, label="Training Set Accuracy")
        plt.plot(testacceval, label="Test Set Accuracy")
        plt.plot(losseval, label="Loss Function")
        plt.legend()

        date_time = datetime.now().strftime("%m_%d_%Y_%Hh%M")
        plt.savefig("./traininggraphs/Results_" + date_time, bbox_inches='tight')
        plt.close()
    # print(mistakes)
    # print("len: ", len(mistakes))
    # print("acc_lens: ", acc_lens)
    # print("acc_cycles: ", acc_cycles)
    # file.write(f"len: {len(mistakes)}\n")
    # file.write("acc_lens: ", acc_lens)
    # file.write("acc_cycles: ", acc_cycles)


def create_and_train_nn(epochs: int, batch_size: int, hiddenchannels: int,
                        trainsrc: str = "", testsrc: str = "", save_training_graphs: bool = True,
                        logging: bool = True, multiple: int = 1) -> None:
    allacs = []
    for i in range(multiple):
        trainset, testset = load_datasets(trainsrc, testsrc)
        model = BA_GCN(hiddenchannels, trainset.num_node_features, trainset.num_classes)
        train_model_and_save_stats(model, epochs, trainset, testset, batch_size, save_training_graphs, logging)
        allacs.append(test(model, DataLoader(testset, batch_size=batch_size, shuffle=True))[0])
    if logging and i > 1:
        log = open("./results/newedgeattrresults.txt", "a")
        log.write(f"Trainset: {trainsrc} Testset: {testsrc} Accuracies: ")
        for a in allacs:
            log.write(str(a*100)+"  ")
        log.write("\n")




EPOCHS = 75
BATCH_SIZE = 125
HIDDEN_CHANNELS = 20
mul = 10

# create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "min1b_250_3_9", "min1b_500_10_25", multiple = mul)
# create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "min1b_1000_3_9", "min1b_500_10_25", multiple = mul)
# create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "min1b_10000_3_9", "min1b_500_10_25", multiple = mul)
create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "min1b_50000_3_9", "min1b_500_10_25", multiple = mul)
#
# create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "infb_250_3_9", "infb_500_10_25", multiple = mul)
# create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "infb_1000_3_9", "infb_500_10_25", multiple = mul)
# create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "infb_10000_3_9", "infb_500_10_25", multiple = mul)
create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "infb_50000_3_9", "infb_500_10_25", multiple = mul)
#
# create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "empty_250_3_9", "empty_500_10_25", multiple = mul)
# create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "empty_1000_3_9", "empty_500_10_25", multiple = mul)
# create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "empty_10000_3_9", "empty_500_10_25", multiple = mul)
create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "empty_50000_3_9", "empty_500_10_25", multiple = mul)


#create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "infb_10000_3_9", "infb_500_10_25")

#create_and_train_nn(EPOCHS, BATCH_SIZE, HIDDEN_CHANNELS, "min1b_1000_3_9", "min1b_500_10_25")
