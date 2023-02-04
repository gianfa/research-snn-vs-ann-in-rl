# %%
"""
3nd Gen. timeseries Classification

Description
-----------
We want to classify the original phases of a given signal.
X -> LIFnet -> y

- Such a signal is a function of two phases w1, w2: X_i = s(w1, w2); w1 << w2
- The labels are such phases, encoded as one-hot arrays: y = 1h(w1, w2)




"""
import sys
sys.path += ["..", "../.."]

import time
import os
from itertools import combinations
from pathlib import Path
from typing import Dict, List

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
from experimentkit.metricsreport import MetricsReport
from experimentkit.generators.time_series import *
from experimentkit.monitor import Monitor
from stdp.funx import stdp_step

DATA_PATH = '../../data'
EXP_PATH = "."
IMGS_PATH = None
EXP_PREFIX = "v1"

imgs_path = IMGS_PATH if IMGS_PATH is not None else Path(EXP_PATH)/"imgs"
os.makedirs(imgs_path, exist_ok=True)

# %% Helper functions


def batch_accuracy(net, data, targets, layer_name=None):
    # From snnTorch examples
    output, _ = net(data.view(batch_size, -1))
    # take idx of the max firing neuron, as the y_pred
    output = output[layer_name] if layer_name else output
    _, idx = output.sum(dim=0).max(dim=1)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc


def print_batch_accuracy(data, targets, train=False):
    acc = batch_accuracy(data, targets)

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


def print_batch_accuracy(data, targets, train=False):
    # https://github.com/jeshraghian/snntorch/blob/master/docs/tutorials/tutorial_5.rst#71-accuracy-metric
    output, _ = net(data.view(batch_size, -1))
    _, idx = output.sum(dim=0).max(1)
    acc = np.mean((targets == idx).detach().cpu().numpy())

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")


def train_printer(
    epoch, iter_counter,
    test_data, test_targets,
    loss_hist, test_loss_hist,
    data, targets,
    ):
    # https://github.com/jeshraghian/snntorch/blob/master/docs/tutorials/tutorial_5.rst#71-accuracy-metric
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(data, targets, train=True)
    print_batch_accuracy(test_data, test_targets, train=False)
    print("\n")


# %% Dataset Creation: Signal definition

batch_size: int = 64

w1 = 2
w2 = w1 * 3
w3 = 4
w4 = w3 * 3
sig_length = 500


dtype = torch.float
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


# Definition of the signal function
sig_2w = lambda w1, w2, t: (1 + np.cos(w1 * t) ) * (1 + np.sin(w2 * t))

ws = [w1, w2, w3, w4]
t = np.arange(sig_length)
signals = [sig_2w(w1, w2, t) for w1, w2 in combinations(ws, 2)]

plot_n_examples(signals, 6)


# Label Encoding
labels = np.arange(len([w1, w2, w3, w4]))

ws_combinations = list(combinations(labels, 2))
ncats = len(ws_combinations)

int2oh = lambda i, ncats: np.identity(ncats)[i, :]
oh2int = lambda oh: np.argwhere(oh).flatten()

labels = ws_combinations
label2oh_dict = {ws_combinations[i]:int2oh(i, ncats) for i in range(ncats)}
label2oh = lambda label: label2oh_dict[label]
oh2label = lambda oh: ws_combinations[oh2int(oh)]

# int2label = {i:ws_combinations[i] for i in range(ncats)}
# label2int = {wsi: i for i, wsi in int2label.items()}

# %% Dataset Creation: batching

def ts_get_batches_and_labels(signals: list, labels: list, batch_size: int):
    """
    Examples
    --------
    >>> nsignals = 3
    >>> signals = np.random.rand(100, nsignals)
    >>> labels = np.arange(nsignals)
    >>> batches, batches_labels = ts_get_batches_and_labels(
    ...     signals, labels, 32)
    """
    batches = []
    batches_labels = []
    for i, si_li in enumerate(zip(signals, labels)):
        si, li = si_li
        sig_i_b = [] # [batch_0_s1 batch_1_s1, ..., batch_n_s1]
        for i in range(0, sig_length, batch_size):
            sig_i_b_i = si[i:min(i + batch_size, len(si))]
            if len(sig_i_b_i) == batch_size:
                sig_i_b.append(sig_i_b_i)
        li_i = [label2oh(li) for _ in sig_i_b]
        batches.append(sig_i_b)
        batches_labels.append(li_i)
    assert len(batches) == len(signals) == len(batches_labels)

    batches = torch.Tensor(np.vstack(batches)).float()
    batches_labels = torch.Tensor(np.vstack(batches_labels)).int()
    assert len(batches) == len(batches_labels)
    return batches, batches_labels

batches, batches_labels = ts_get_batches_and_labels(
    signals, labels, batch_size)
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

# train/valid/test split
total_count = len(signal_ds)
train_count = int(0.7 * total_count)
valid_count = int(0.2 * total_count)
test_count = total_count - train_count - valid_count
train_dataset, valid_dataset, test_dataset = torch.utils.data.random_split(
    signal_ds, (train_count, valid_count, test_count)
)

train_dl = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
valid_dl = DataLoader(valid_dataset, shuffle=True, batch_size=batch_size)
test_dl = DataLoader(test_dataset, shuffle=True, batch_size=batch_size)

# # Check
# for x_i, y_i in train_dl:
#     print(x_i.shape, y_i.shape)


# %% # Network Definition

# Network Architecture
num_inputs = batch_size
num_hidden = 10
num_outputs = ncats # n combinations

num_steps = 5
beta = 0.95
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        # Initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta)
        self.lif1.name = 'lif1'
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta)
        self.lif2.name = 'lif2'

        self.lif_layers_names = ['lif1', 'lif2']
        self.spk_rec = {nm: [] for nm in self.lif_layers_names}
        self.mem_rec = {nm: [] for nm in self.lif_layers_names}

    def forward(self, x):

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        # Record the layers
        #    These will collect the spiking history for all the
        #    training steps
        spk_rec_fw = {nm: [] for nm in self.lif_layers_names}
        mem_rec_fw = {nm: [] for nm in self.lif_layers_names}

        # Record the layers during inner (spiking) steps
        for _ in range(num_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk_rec_fw['lif1'].append(spk1)
            mem_rec_fw['lif1'].append(mem1)
            spk_rec_fw['lif2'].append(spk2)
            mem_rec_fw['lif2'].append(mem2)

        spk_out = torch.stack(spk_rec_fw['lif2'], dim=0)
        mem_out = torch.stack(mem_rec_fw['lif2'], dim=0)

        # Append the new batch of inner steps to the recs
        # self.spk_rec [=] (examples, num_steps, num_outputs)
        for nm in self.lif_layers_names:
            self.spk_rec[nm].append(torch.stack(spk_rec_fw[nm], dim=0))
        for nm in self.lif_layers_names:
            self.mem_rec[nm].append(torch.stack(mem_rec_fw[nm], dim=0))
        
        return spk_out, mem_out



# Load the network onto CUDA if available
net = Net().to(device)


loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))


# %% # Training Loop

num_epochs = 200

train_mr = MetricsReport()
valid_mr = MetricsReport()
loss_hist = []
valid_loss_hist = []

counter = 0
t_start = time.time()
dt_stdp = 10

get_model_layer_params = lambda layer_name: \
    dict(net.named_parameters())[f'{layer_name}.weight']

# container structures
weights = {
    'fc2': [dict(net.named_parameters())[f'fc2.weight'].clone()]
}
stdp_dt_idxs = []
spk2_rec = []
mem2_rec = []
counter = 0

# loss_monitor = Monitor(
#     x=range(len(loss_hist)), y=loss_hist, plot_kwargs={'marker': 'x'},
#     title="Loss", xlabel="# iterations", pause=0.01)
# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0

    # Minibatch training loop
    for Xi, yi in iter(train_dl):
        Xi = Xi.to(device)
        yi = yi.to(device)

        net.train()
        # mem_out [=] (num_steps, ?, ncats)
        spk_out, mem_out = net(Xi)
        print(f"spk_out.shape: {spk_out.shape}")
        # Let's take the max membrane at time step, as the most probable cat
        yi_pred = mem_out.max(dim=0)[0]

        # Loss
        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val = loss_val + loss_fn(mem_out[step], yi.float())

        # Gradient calculation + weight update
        optimizer.zero_grad() # prepare for the step
        loss_val.backward() # propagate loss gradient
        optimizer.step() # update the parameters

        print(type(yi), type(yi_pred))
        # train_mr.append(oh2int(yi.flatten()).item, yi_pred.item())
        iter_results = {
            'epoch': epoch,
            'iteration': iter_counter,
            'loss': f"{loss_val.item():.2f}",
            'b. accuracy': f"{torch.sum(yi_pred==yi)/yi_pred.shape[0]:.2f}"
        }
        print_vars_table(
            iter_results,iter_counter % 50 == 0 or iter_counter == 0)

        # Store loss history for future plotting
        loss_hist.append(loss_val.item())

        # Validation set
        with torch.no_grad():
            net.eval()
            Xi_valid, yi_valid = next(iter(valid_dl))
            Xi_valid = Xi_valid.to(device)
            yi_valid = yi_valid.to(device)

            # Valid set forward pass
            spk_out, mem_out = net(Xi)
            yi_pred = mem_out.max(dim=0)[0]

            # Valid set loss
            loss_valid = torch.zeros((1), dtype=dtype, device=device)
            for step in range(num_steps):
                loss_valid = loss_valid + loss_fn(mem_out[step], yi.float())
            
            # _, train_y_pred = torch.max(output.data, 1)
            # train_mr.append(targets, train_y_pred)
            valid_loss_hist.append(loss_valid.item())

            # Print train/test loss/accuracy
            # if counter % 50 == 0:
            #     train_printer(
            #         data, targets, epoch, counter, iter_counter,
            #         loss_hist, valid_loss_hist, Xi_valid, yi_valid)
            counter += 1
            iter_counter += 1

print(f"Elapsed time: {int(time.time() - t_start)}s")

# %%
# Visualize
# Plot Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(valid_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Test Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

# %%
