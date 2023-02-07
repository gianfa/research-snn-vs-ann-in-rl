# %%
"""
3nd Gen. timeseries Classification

Description
-----------
We want to classify the original phases of a given signal.
X -> LIFnet -> y

- Such a signal is a function of two phases w1, w2: X_i = s(w1, w2); w1 << w2
- The labels are such phases, encoded as one-hot arrays: y = 1h(w1, w2)

ISSUE:
- Being the test set very similar (or equal) to the training set, here we are
    in a data-leak-like situation 



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
from sklearn.metrics import ( # noqa
    accuracy_score, precision_score, f1_score, confusion_matrix)
import torch  # noqa
import torch.nn as nn  # noqa
from torch.utils.data import Dataset, DataLoader, TensorDataset  # noqa
from torchvision import datasets, transforms  # noqa

from experimentkit.visualization import plot_n_examples
from experimentkit.metricsreport import MetricsReport
from experimentkit.metricsreport import MetricsReport
from experimentkit.generators.time_series import *
from experimentkit.monitor import Monitor
from experimentkit.funx import pickle_save_dict
from stdp.funx import stdp_step

DATA_PATH = '../../data'
EXP_PREFIX = "hidden_16-td_50"
EXP_PATHS = {}
EXP_PATHS['root'] = Path(".")
EXP_PATHS['imgs'] = EXP_PATHS['root']/"imgs"
EXP_PATHS['data'] = EXP_PATHS['root']/"data"

imgs_path = EXP_PATHS['imgs']
for dir_name, dir_path in EXP_PATHS.items():
    os.makedirs(dir_path, exist_ok=True)

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

batch_size: int = 32

w1 = 2
w2 = w1 * 3
w3 = 4
w4 = w3 * 3

sig_length = 40 # length of the Xi signals
tot_sig_length = 20 * 40 # length of the parent sig, to be cut to have many Xi



dtype = torch.float
device = (
    torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))


# Definition of the signal function
sig_2w = lambda w1, w2, t: (1 + np.cos(w1 * t) ) * (1 + np.sin(w2 * t))

ws = [w1, w2, w3, w4]
t = np.arange(tot_sig_length)
parent_signals = [sig_2w(w1, w2, t) for w1, w2 in combinations(ws, 2)]

plot_n_examples(parent_signals, 6)


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

def ts_get_signals_and_labels(
        parent_signals: list, labels: list, sig_length: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    
    It discards the last batch if its length is lower than sig_length

    Examples
    --------
    >>> nsignals = 3
    >>> signals = np.random.rand(100, parent_signals)
    >>> labels = np.arange(nsignals)
    >>> signals, signals_labels = ts_get_signals_and_labels(
    ...     signals, labels, 32)
    """
    signals = []
    signals_labels = []
    for i, psi_li in enumerate(zip(parent_signals, labels)):
        psi, li = psi_li # parent_signal_i, label_i
        sig_i_b = [] # [batch_0_s1 batch_1_s1, ..., batch_n_s1]
        for i in range(0, len(psi), sig_length):
            sig_i_b_i = psi[i:min(i + sig_length, len(psi))]
            if len(sig_i_b_i) == sig_length:
                sig_i_b.append(sig_i_b_i)
        li_i = [label2oh(li) for _ in sig_i_b]
        signals.append(sig_i_b)
        signals_labels.append(li_i)
    assert len(signals) == len(signals) == len(signals_labels)

    signals = torch.Tensor(np.vstack(signals)).float()
    signals_labels = torch.Tensor(np.vstack(signals_labels)).int()
    assert len(signals) == len(signals_labels)
    return signals, signals_labels

signals, signals_labels = ts_get_signals_and_labels(
    parent_signals, labels, sig_length=sig_length)

plot_n_examples(signals, 6)
print(
    f"signals.shape: {signals.shape},\n"+
    f"signals_labels.shape: {signals_labels.shape}")
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

signal_ds = BiPhasicDataset(signals, signals_labels)
signal_dl = DataLoader(signal_ds, batch_size=batch_size)

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

print(
    f"Train len:{len(train_dl.dataset)}\n" +
    f"valid len:{len(valid_dl.dataset)}\n" +
    f"test len:{len(test_dl.dataset)}\n")
# # Check
# for x_i, y_i in train_dl:
#     print(x_i.shape, y_i.shape)


# %% # Network Definition

# Network Architecture
num_inputs = sig_length
num_hidden = 1
num_outputs = ncats # n combinations

n_net_inner_steps = 50
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
        for _ in range(n_net_inner_steps):
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
        # self.spk_rec [=] (examples, n_net_inner_steps, num_outputs)
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

weights_hist = [weights]
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
        print(f"c: {counter}")
        Xi = Xi.to(device) # [=] (batch_size, sig_length)
        yi = yi.to(device) # [=] (batch_size, ncats)

        net.train()
        # mem_out [=] (n_net_inner_steps, batch_size, ncats)
        spk_out, mem_out = net(Xi)

        # Let's take the max membrane at time step, as the most probable cat
        yi_pred = mem_out.max(dim=0)[0] # [=] (batch_size, ncats)
        yi_pred_int = yi_pred.argmax(dim=1)
        yi_int = yi.argwhere()[:, 1] # form 1-h

        # Loss
        # initialize the loss & sum over time
        # Here the loss is a batch loss (1 per batch)
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(n_net_inner_steps):
            loss_val = loss_val + loss_fn(mem_out[step], yi.float())

        # Gradient calculation + weight update
        optimizer.zero_grad() # prepare for the step
        loss_val.backward() # propagate loss gradient
        optimizer.step() # update the parameters

        # STDP #
        # CAVEAT: spk_out is already a collection of many iterations
        if counter > 0 and counter % dt_stdp == 0:
            steps = (counter - dt_stdp, counter)
            # cur_raster [=] (dt_stdp, batch_size, ncats)
            cur_raster = spk_out[steps[0]:steps[1]]
            weights = dict(net.named_parameters())['fc2.weight']
            new_weights=stdp_step(
                weights=weights,
                connections=None,
                raster=cur_raster,
                # spike_collection_rule=all_to_all,
                # dw_rule="sum",
                bidirectional=True,
                max_delta_t=20,
                inplace=False,
                v=False,
            )
            # Update the Net weights
            for name, param in net.named_parameters():
                if name == "fc2.weight":
                    param.data = nn.parameter.Parameter(new_weights.clone())
                    break
            assert torch.allclose(dict(net.named_parameters())['fc2.weight'], new_weights)
            # store the new weights
            stdp_dt_idxs.append(step)
        weights_hist.append(weights)

        train_mr.append(yi_int, yi_pred_int)
        iter_results = {
            'epoch': epoch,
            'iteration': iter_counter,
            'loss': f"{loss_val.item():.2f}",
            'b. accuracy': \
                f"{torch.sum(yi_pred_int==yi_int)/yi_pred_int.shape[0]:.2f}"
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
            yi_valid_pred = mem_out.max(dim=0)[0]
            yi_valid_pred_int = yi_valid_pred.argmax(dim=1)
            yi_valid_int = yi.argwhere()[:, 1] # form 1-h

            # Valid set loss
            loss_valid = torch.zeros((1), dtype=dtype, device=device)
            for step in range(n_net_inner_steps):
                loss_valid = loss_valid + loss_fn(mem_out[step], yi.float())
            
            # _, train_y_pred = torch.max(output.data, 1)
            valid_mr.append(yi_valid_int, yi_valid_pred_int)
            valid_loss_hist.append(loss_valid.item())

            # Print train/test loss/accuracy
            # if counter % 50 == 0:
            #     train_printer(
            #         data, targets, epoch, counter, iter_counter,
            #         loss_hist, valid_loss_hist, Xi_valid, yi_valid)
        counter += 1
        iter_counter += 1

training_time = int(time.time() - t_start)
print(f"Elapsed time: {int(time.time() - t_start)}s")
# %% Visualize: Plot Train/Valid Loss
fig = plt.figure(facecolor="w", figsize=(10, 5))
plt.plot(loss_hist)
plt.plot(valid_loss_hist)
plt.title("Loss Curves")
plt.legend(["Train Loss", "Valid Loss"])
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()

fig.savefig(imgs_path/f"{EXP_PREFIX}-Loss.png")

# %% Evaluation: Test Set

test_mr = MetricsReport()
test_loss_hist = []
for Xi_test, yi_test in iter(test_dl):
    Xi_test = Xi_test.to(device) # [=] (batch_size, sig_length)
    yi_test = yi_test.to(device) # [=] (batch_size, ncats)

    with torch.no_grad():
        # mem_out [=] (n_net_inner_steps, batch_size, ncats)
        spk_out, mem_out = net(Xi_test)

        # Let's take the max membrane at time step, as the most probable cat
        yi_test_pred = mem_out.max(dim=0)[0] # [=] (batch_size, ncats)
        yi_test_pred_int = yi_test_pred.argmax(dim=1)
        yi_test_int = yi_test.argwhere()[:, 1] # form 1-h

        # Test set loss
        # Here the loss is a batch loss
        loss_test = torch.zeros((1), dtype=dtype, device=device)
        for step in range(n_net_inner_steps):
            loss_test = loss_test + loss_fn(mem_out[step], yi_test.float())
        
        # _, train_y_pred = torch.max(output.data, 1)
        test_mr.append(yi_test_int, yi_test_pred_int)
        test_loss_hist.append(loss_test.item())

# %% Visualize: Plot Test Loss
if len(test_loss_hist) > 1:
    fig = plt.figure(facecolor="w", figsize=(10, 5))
    plt.plot(test_loss_hist)
    plt.title("Test Loss")
    plt.legend(["Test Loss"])
    plt.xlabel("Iteration")
    plt.ylabel("Loss")
    plt.show()

    fig.savefig(imgs_path/f"{EXP_PREFIX}-Test-Loss.png")

print(test_mr.get_metrics(return_frame=True))

# %%

cm_ax = train_mr.plot_confusion_matrix(title="Train ConfMat")
cm_ax.figure.savefig(imgs_path/f"{EXP_PREFIX}-ConfMat-train.png")

valid_mr.plot_confusion_matrix(title="Valid ConfMat")

# %% Store project metadata

proj_desc = {
    'label': '',
    'ws': [ws],
    'sig_length': sig_length,
    'train_dl-len': len(train_dl.dataset),
    'batch_size': batch_size,
    'num_epochs': num_epochs,
    'net_inner_delta_t': n_net_inner_steps,
    'train-metrics': train_mr.get_metrics(return_frame=True).tail(1).to_dict(),
    'training-time': training_time,
    'test-metrics': test_mr.get_metrics(return_frame=True),
    'net': {
        'neuron_beta': beta
    },
    'description': ""
}
pickle_save_dict(EXP_PATHS['data']/f"{EXP_PREFIX}-metadata.pkl", proj_desc)
proj_desc

# %%