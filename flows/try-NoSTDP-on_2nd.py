# %%
"""Implementation trials, on 2ndGen, for comparison

check try-STDP-on_LIF-training.py

Decription
----------



References
----------
1. https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html#define-the-network
2. https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_6.html
3. `NEFTCI, Emre O.; MOSTAFA, Hesham; ZENKE, Friedemann. Surrogate gradient learning in spiking neural networks: Bringing the power of gradient-based optimization to spiking neural networks. IEEE Signal Processing Magazine, 2019, 36.6: 51-63. <https://core.ac.uk/download/pdf/323310555.pdf>`_


Theory
------
Learning: Loss
- Surrogate gradient approach.
- Spike-Operator approach. -> ∂S/∂U ϵ {0,1}


Output decoding
- we take the output layer as a logit layer, where each neuron
    fires for a specific class.
- we choose "rate coding" as a decoding strategy. This means that
    we'll expect the neuron assigned to the right class to fire
    at higher freq during the specific class showing up.
    - ONE way is to increase U_th of the neuron and decrease the others ones
        - softmax p_i[t] = exp(U_i[t] / sum( exp(U_j[t]), 0, C))
        - L_{CE}[t] = sum( y_i log(p_i[t]), 0,C )
        - => L_{CE} = sum( L_{CE}[t] ,t)

"""

# %% Imports

import sys
sys.path.append("..")

import time
import os
from typing import Dict, List
import seaborn as sns
import snntorch as snn  # noqa
from snntorch import spikeplot as splt  # noqa
from snntorch import spikegen  # noqa
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix  # noqa

import torch  # noqa
import torch.nn as nn  # noqa
from torch.utils.data import DataLoader  # noqa
from torchvision import datasets, transforms  # noqa

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd # noqa
import itertools  # noqa

from experimentkit.metricsreport import MetricsReport

# %% Helper functions


def batch_accuracy(net, data, targets):
    output = net(data.view(batch_size, -1))
    # take idx of the max firing neuron, as the y_pred
    _, idx = output.sum(dim=0).max(dim=1)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc


def print_batch_accuracy(net, data, targets, train=False):
    acc = batch_accuracy(net, data, targets)

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def print_vars_table(arg_dict: dict, print_title: bool=False) -> None:
    if print_title:
        title = "|" + "|".join(
            [f"{arg_name:^10}" for arg_name in arg_dict.keys()]) + "|"
        title += "\n|" + "|".join(
            [f"{'----------':^10}" for arg_name in arg_dict.keys()]) + "|"
        print(title)

    values = "|" + "|".join(
            [f"{arg_value:^10}" for arg_value in arg_dict.values()]) + "|"
    print(values)
    return None


def train_printer(
    data, targets,
    epoch, counter, iter_counter, loss_hist,
        test_loss_hist, test_data, test_targets):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")



# %% Data Loading

# dataloader arguments
batch_size = 128
data_path = '../data'


if not os.path.isdir(data_path):
    raise Exception("Data directory not found")

dtype = torch.float
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))

# Define a transform
transform = transforms.Compose([
            transforms.Resize((28, 28)),
            transforms.Grayscale(),
            transforms.ToTensor(),
            transforms.Normalize((0,), (1,))])

mnist_train = datasets.MNIST(
    data_path, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(
    data_path, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(
    mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


# %% # Network Definition

# Network Architecture
num_inputs = 28 * 28
num_hidden = 100
num_outputs = 10


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.act2 = nn.Tanh()

    def forward(self, x):
        x = self.fc1(x)
        x = self.act1(x)
        x = self.fc2(x)
        x = self.act2(x)
        return x


# Load the network onto CUDA if available
net = Net().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))

"""
    After only one iteration, the loss should have decreased and accuracy
should have increased.
"""


# %% # Training Loop

num_epochs = 1

train_mr = MetricsReport()
valid_mr = MetricsReport()
loss_hist = []
test_loss_hist = []

counter = 0
t_start = time.time()
# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    for data, targets in train_batch:
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad() # prepare for the step

        # Forward
        output = net(data.view(batch_size, -1)).float()
        # Loss
        _, y_pred = torch.max(output.data, 1)
        loss = loss_fn(output, targets)
        # loss.requires_grad = True
        # Backward
        loss.backward() # propagate loss gradient

        optimizer.step() # update the parameters

        train_mr.append(targets, y_pred)
        iter_results = {
            'epoch': epoch,
            'iteration': iter_counter,
            'loss': f"{loss:.2f}",
            'b. accuracy': f"{torch.sum(y_pred==targets)/y_pred.shape[0]:.2f}"
        }
        print_vars_table(
            iter_results,iter_counter % 50 == 0 or iter_counter == 0)

        # Store loss history for future plotting
        loss_hist.append(loss.item())

        # Validation set
        with torch.no_grad():
            net.eval()
            test_data, test_targets = next(iter(test_loader))
            test_data = test_data.to(device)
            test_targets = test_targets.to(device)

            # Test set forward pass
            test_output = net(test_data.view(batch_size, -1))

            # Test set loss
            test_loss = loss_fn(test_output, test_targets)
            
            _, train_y_pred = torch.max(output.data, 1)
            train_mr.append(targets, train_y_pred)
            test_loss_hist.append(test_loss.item())

            # Print train/test loss/accuracy
            # if counter % 50 == 0:
            #     train_printer(
            #         data, targets, epoch, counter, iter_counter,
            #         loss_hist, test_loss_hist, test_data, test_targets)
            counter += 1
            iter_counter += 1

print(f"Elapsed time: {int(time.time() - t_start)}s")

# %%
# Visualize
# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(test_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

#  Evaluation on Test
total = 0
correct = 0

# drop_last switched to False to keep all samples
# test_loader = DataLoader(
#     mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

# with torch.no_grad():
#     net.eval()
#     for data, targets in test_loader:
#         data = data.to(device)
#         targets = targets.to(device)

#         # forward pass
#         test_spk, _ = net(data.view(data.size(0), -1))

#         # calculate total accuracy
#         _, predicted = test_spk.sum(dim=0).max(1)
#         total += targets.size(0)
#         correct += (predicted == targets).sum().item()

# print(f"Total correctly classified test set images: {correct}/{total}")
# print(f"Test Set Accuracy: {100 * correct / total:.2f}%")


ms, ms_ns = train_mr.get_metrics(return_names=True)
metrics_hist_df = pd.DataFrame(ms, columns=ms_ns)
metrics_hist_df
# %%

train_mr.plot_confusion_matrix(
    title='train Confusion Matrix', fmt=f".1f", normalize='true')
# valid_mr.plot_confusion_matrix(
#     title='valid Confusion Matrix', fmt=f".1f", normalize='true')
# %%
