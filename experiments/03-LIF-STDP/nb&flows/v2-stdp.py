# %%
""" [INCOMPLETE] STDP Implementation trials, on LIFs

NOTES
-----
IN-DEVELOPMENT. This exp is not ready yet


Description
----------
Contains tests of STDP done on LIF networks.


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
sys.path += ["..", "../..", "../../.."]

import time
import os
from pathlib import Path
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

from experimentkit_in.metricsreport import MetricsReport
from stdp.funx import stdp_step

# Project Parameters #
ROOT_DIR = Path("../../../")
DATA_DIR = ROOT_DIR/"data"
EXP_DIR = ROOT_DIR/"experiments/LIF-STDP"
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'





# %% Helper functions


def batch_accuracy(net, data, targets, layer_name=None):
    # From snnTorch examples
    output, _ = net(data.view(batch_size, -1))
    # take idx of the max firing neuron, as the y_pred
    output = output[layer_name] if layer_name else output
    _, idx = output.sum(dim=0).max(dim=1)
    acc = np.mean((targets == idx).detach().cpu().numpy())
    return acc


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


def print_batch_accuracy(net, data, targets, train=False, layer_name=None):
    acc = batch_accuracy(net, data, targets, layer_name)

    if train:
        print(f"Train set accuracy for a single minibatch: {acc*100:.2f}%")
    else:
        print(f"Test set accuracy for a single minibatch: {acc*100:.2f}%")

def train_printer(
    net,
    epoch, iter_counter,
    test_data, test_targets,
    loss_hist, test_loss_hist,
    data, targets,
    counter,
    layer_name=None):
    print(f"Epoch {epoch}, Iteration {iter_counter}")
    print(f"Train Set Loss: {loss_hist[counter]:.2f}")
    print(f"Test Set Loss: {test_loss_hist[counter]:.2f}")
    print_batch_accuracy(net, data, targets, train=True, layer_name=layer_name)
    print_batch_accuracy(net, 
        test_data, test_targets, train=False, layer_name=layer_name)
    print("\n")



# %% Data Loading

# dataloader arguments
batch_size = 32


if not os.path.isdir(DATA_DIR):
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
    DATA_DIR, train=True, download=True, transform=transform)
mnist_test = datasets.MNIST(
    DATA_DIR, train=False, download=True, transform=transform)

# Create DataLoaders
train_loader = DataLoader(
    mnist_train, batch_size=batch_size, shuffle=True, drop_last=True)
test_loader = DataLoader(
    mnist_test, batch_size=batch_size, shuffle=True, drop_last=True)


# %% # Network Definition

# Network Architecture
num_inputs = 28 * 28
num_hidden = 200
num_outputs = 10

# Temporal Dynamics
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

        # Record the layers
        #    These will collect the spiking history for all the
        #    training steps
        # Spikes/activations Recorder
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
        for nm in self.lif_layers_names:
            self.spk_rec[nm].append(torch.stack(spk_rec_fw[nm], dim=0))
        for nm in self.lif_layers_names:
            self.mem_rec[nm].append(torch.stack(mem_rec_fw[nm], dim=0))
        
        return spk_out, mem_out


# Load the network onto CUDA if available
net = Net().to(device)


loss = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-4, betas=(0.9, 0.999))


# %% # Training Loop

num_epochs = 1

train_mr = MetricsReport()
valid_mr = MetricsReport()
loss_hist = []
test_loss_hist = []

counter = 0
t_start = time.time()
dt_stdp = 2

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
# Outer training loop
for epoch in range(num_epochs):
    iter_counter = 0
    train_batch = iter(train_loader)

    # Minibatch training loop
    # for data, targets in train_batch:
    for i, data_targets in enumerate(train_batch):
        data, targets = data_targets
        data, targets = data, targets
        data = data.to(device)
        targets = targets.to(device)

        # forward pass
        net.train()
        # mem_out [=] (num_steps, batch_size, num_outputs)
        spk_out, mem_out = net(data.view(batch_size, -1))

        # initialize the loss & sum over time
        loss_val = torch.zeros((1), dtype=dtype, device=device)
        for step in range(num_steps):
            loss_val = loss_val + loss(mem_out[step], targets.long())
        

        # take idx of the max firing neuron, as the y_pred
        #   spk_out [=] batch_size x t_delta
        # _, y_pred = spk_out.sum(dim=0).max()
        # train_mr.append(targets, y_pred)
        # iter_results = {
        #     'epoch': epoch,
        #     'iteration': iter_counter,
        #     'loss': f"{loss_val.item():.2f}",
        #     'b. accuracy': f"{batch_accuracy(net, data, targets, 'lif2'):.2f}"
        # }
        # print_vars_table(
        #     iter_results,iter_counter % 50 == 0 or iter_counter == 0)

        # Gradient calculation + weight update
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step() 

        # STDP #
        # CAVEAT: spk_out is already a collection of many iterations
        if i > 0 and i % dt_stdp == 0:
            steps = (i - dt_stdp, i)
            # cur_raster = raster[steps[0]:steps[1]]
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
            for name, param in net.named_parameters():
                if name == "fc2.weight":
                    param.data = nn.parameter.Parameter(new_weights.clone())
            assert dict(net.named_parameters())['fc2.weight'] == new_weights
            # store the new weights
            weights['fc2'].append(new_weights.clone())
            stdp_dt_idxs.append(step)


        # Store loss history for future plotting
        # loss_hist.append(loss_val.item())

        # Validation set
        # with torch.no_grad():
        #     net.eval()
        #     test_data, test_targets = next(iter(test_loader))
        #     test_data = test_data.to(device)
        #     test_targets = test_targets.to(device)

        #     # Test set forward pass
        #     test_spk, test_mem = net(test_data.view(batch_size, -1))

        #     # Test set loss
        #     test_loss = torch.zeros((1), dtype=dtype, device=device)
        #     for step in range(num_steps):
        #         test_loss += loss(test_mem[step], test_targets)
        #     test_loss_hist.append(test_loss.item())

        #     _, test_y_pred = test_spk.sum(dim=0).max(dim=1)
        #     valid_mr.append(targets, test_y_pred)

        #     # Print train/test loss/accuracy
        #     if counter % 50 == 0:
        #         train_printer( net,
        #             data, targets, epoch, counter, iter_counter,
        #             loss_hist, test_loss_hist, test_data, test_targets, layer_name='lif2')
        #     counter += 1
        iter_counter += 1

print(f"Elapsed time: {int(time.time() - t_start)}s")
print(f"{len(stdp_dt_idxs)} STDP step performed")


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
test_loader = DataLoader(
    mnist_test, batch_size=batch_size, shuffle=True, drop_last=False)

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

# %%

ms, ms_ns = train_mr.get_metrics(return_names=True)
metrics_hist_df = pd.DataFrame(ms, columns=ms_ns)
metrics_hist_df

train_mr.plot_confusion_matrix(
    title='train Confusion Matrix', fmt=f".1f", normalize='true')
valid_mr.plot_confusion_matrix(
    title='valid Confusion Matrix', fmt=f".1f", normalize='true')
# %%
