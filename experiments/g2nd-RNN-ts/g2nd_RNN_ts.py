# %%
"""
2nd Gen. timeseries classification

Description
-----------
We want to classify the original phases of a given signal.
X -> RNN -> y

Such a signal is a function of w1, w2, phases
X_i = s(w1, w2)
w1 << w2
The label is a tuple of these phases
y = (w1, w2)


Assumptions
-----------
- in a ESN with plasticity, the UL happens automatically while the spikes are
    propagated through the network, up to the last neurons, if there are any by
    design, otherwise untill a stop rule is applied.
- We then leverage a decoder in order to understand the response of the
    network to specific inputs.



"""
import sys
sys.path.append("../..")

import time
import os
from typing import Dict, List
from itertools import combinations

import matplotlib.pyplot as plt  # noqa
import numpy as np  # noqa
import pandas as pd # noqa

import seaborn as sns
import snntorch as snn  # noqa
from snntorch import spikeplot as splt  # noqa
from snntorch import spikegen  # noqa
from sklearn.metrics import accuracy_score, precision_score, f1_score, confusion_matrix  # noqa

import torch  # noqa
import torch.nn as nn  # noqa
from torch.utils.data import Dataset, DataLoader, TensorDataset  # noqa
from torchvision import datasets, transforms  # noqa

from experimentkit.visualization import plot_n_examples
from experimentkit.metricsreport import MetricsReport

# %% Helper functions


def batch_accuracy(net, data, targets, batch_size):
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


# %% Dataset Creation: Signal definition

batch_size = 64
data_path = '../../data'

w1 = 2
w2 = w1 * 7e3
w3 = 31
w4 = w1 * 5e3
sig_length = 1000


if not os.path.isdir(data_path):
    raise Exception("Data directory not found")

dtype = torch.float
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


# Definition of the signal function
sig_2w = lambda w1, w2, t: (1 + np.cos(w1 * t) ) * (1 + np.sin(w2 * t))

ws = [w1, w2, w3, w4]
t = np.arange(sig_length)
signals = [sig_2w(w1, w2, t) for w1, w2 in combinations(ws, 2)]

# Label Encoding
labels = np.arange(len([w1, w2, w3, w4]))
ws_combinations = list(combinations(labels, 2))
ncats = len(ws_combinations)

labels = ws_combinations

plot_n_examples(signals, 6)
# %% Dataset Creation: batching

batches = []
batches_labels = []
for i, si_li in enumerate(zip(signals, labels)):
    si, li = si_li
    sig_i_b = [] # [batch_0_s1 batch_1_s1, ..., batch_n_s1]
    for i in range(0, sig_length, batch_size):
        sig_i_b_i = si[i:min(i + batch_size, len(si))]
        if len(sig_i_b_i) == batch_size:
            sig_i_b.append(sig_i_b_i)
    li_i = [li for _ in sig_i_b]
    batches.append(sig_i_b)
    batches_labels.append(li_i)

assert len(batches) == len(signals)
batches = torch.Tensor(np.vstack(batches)).float()
batches_labels = torch.Tensor(np.vstack(batches_labels))

plot_n_examples(batches, 6)
# %% Dataset Creation: load into DataLoaders

class BiPhasicDataset(Dataset):
    def __init__(self, X, y):
        assert len(X) == len(y)
        super().__init__()
        self.X = X
        self.y = y

    def __getitem__(self, i):
        return self.X[i], self.y[i]

    def __len__(self):
        return len(self.X)

signal_ds = BiPhasicDataset(batches, batches_labels)
signal_dl = DataLoader(signal_ds)


total_count = len(signal_ds)
train_count = int(0.7 * total_count)
valid_count = int(0.2 * total_count)
test_count = total_count - train_count - valid_count
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    signal_ds, (train_count, valid_count, test_count)
)

train_dl = DataLoader(train_dataset)

# # Check
# for x_i, y_i in train_dl:
#     print(x_i.shape, y_i.shape)


# %% # Network Definition

# Network Architecture
input_size = batch_size
hidden_size = 100
output_size = ncats # n combinations


class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.rnn1 = nn.RNN(
            input_size=input_size, hidden_size=hidden_size, num_layers=1)
        self.act1 = nn.Tanh()
        self.fc2 = nn.Linear(hidden_size, output_size)


    def forward(self, x):
        x, h_n = self.rnn1(x)
        x = self.act1(x)
        x = self.fc2(x)
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

    # Minibatch training loop
    for data, targets in iter(train_dl):
        data = data.to(device)
        targets = targets.to(device)

        optimizer.zero_grad() # prepare for the step

        # Forward
        output = net(data.float())
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
        # with torch.no_grad():
        #     net.eval()
        #     test_data, test_targets = next(iter(test_dl))
        #     test_data = test_data.to(device)
        #     test_targets = test_targets.to(device)

        #     # Test set forward pass
        #     test_output = net(test_data.view(batch_size, -1))

        #     # Test set loss
        #     test_loss = loss_fn(test_output, test_targets)
            
        #     _, train_y_pred = torch.max(output.data, 1)
        #     train_mr.append(targets, train_y_pred)
        #     test_loss_hist.append(test_loss.item())

        #     # Print train/test loss/accuracy
        #     # if counter % 50 == 0:
        #     #     train_printer(
        #     #         data, targets, epoch, counter, iter_counter,
        #     #         loss_hist, test_loss_hist, test_data, test_targets)
        #     counter += 1
        #     iter_counter += 1

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

