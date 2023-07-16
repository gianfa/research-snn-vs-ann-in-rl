""" Training a Network on Lorenz data

Forecasting: Y = X[t+dt]

1. A dataset is generated from chaotic Lorenz System.
2. A SNN is created as a Input + LIF + LIF.
    1. Each LIF has not trainable units
    2. The network has trainable parameters (Linear layers)
3. The input examples to the network are mini batches of dims (batch_size, 3),
    while the output labels are (batch_size, 3).



NOTES
-----
At the moment seems like the network is not learning.
The Loss appears too much periodic and not asymptotically going to zero.


References
----------
1. https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html
"""
# %%
from functools import partial
import sys

import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import snntorch as snn
from sklearn.model_selection import train_test_split


sys.path += ['.', '../', '../..', '../../../']
from experimentkit_in.funx import pickle_save_dict, pickle_load
from experimentkit_in.generators.time_series import gen_lorenz
import src09

import wandb
wandb.login()
wandb.init(project='snnTorch-Lorenz-1', entity='gianfa')

ROOT = Path('../../../')
DATA_DIR = ROOT/'data'
EXP_DIR = ROOT/'experiments/09-LIF-understand'
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'

assert EXP_DIR.exists(), \
    f"CWD: {Path.cwd()}\nROOT: {ROOT.absolute()}"

# %% Data Loading

ds_path = EXP_DATA_DIR/"ds_lorenz.pkl"
if  not ds_path.exists():
    ds = gen_lorenz(n_steps=10_000, s=12, r=30, b=2.700)
    pickle_save_dict(ds_path, ds)
else:
    ds = pickle_load(ds_path)


shift = 10
X = ds[:-shift]
y = ds[shift:]

print(X.shape, y.shape)
assert X.shape[0] == y.shape[0]

# %% Train/Test split

test_size = 0.2
valid_size = 0.15

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, shuffle=False)

valid_size = 0.15 / (1 - test_size)
X_train, X_valid, y_train, y_valid = \
    train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

X_train = torch.tensor(X_train, dtype=float)
X_valid = torch.tensor(X_valid, dtype=float)
X_test = torch.tensor(X_test, dtype=float)
y_train = torch.tensor(y_train, dtype=float).unsqueeze(1)
y_valid = torch.tensor(y_valid, dtype=float).unsqueeze(1)
y_test = torch.tensor(y_test, dtype=float).unsqueeze(1)

print(
    X_train.shape, y_train.shape,
    X_valid.shape, y_valid.shape,
    X_test.shape, y_test.shape)


# %% Prepare the containers for recording mem pot and spike hisotries

mem_rec = {  # will collect membrane potentials
    'lif1': [],
    'lif2': [],
}
spk_rec = {  # will collect spikes
    'lif1': [],
    'lif2': [],
}

def record_outputs(layer, input, output, label: str):
    spk, mem = output
    mem_rec[label].append(mem)
    spk_rec[label].append(spk)



# %% Define the Network

class Net(nn.Module):
    def __init__(
            self, num_inputs, num_hidden, num_outputs, beta1=0.5, beta2=0.5):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta1, threshold=15)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2, threshold=10)
        self.inner_steps = 10

    def forward(self, x):
        """
        At each step the x_i example is shown to the network.
        For all the internal dynamics execution, namely during all the 
        internal loop, the same x_i is shown to the network.
        """

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(self.inner_steps):

            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(mem1)
            spk2, mem2 = self.lif2(cur2, mem2)

        return spk2, mem2

# %% Training

# exp params
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Small step current input


bs = 50  # batch size
X_train_mb = X_train[:X_train.shape[0]//bs * bs].reshape(
    X_train.shape[0]//bs, bs, X_train.shape[-1]).to(dtype)
y_train_mb = y_train[:y_train.shape[0]//bs * bs].reshape(
    y_train.shape[0]//bs, bs, y_train.shape[-1]).to(dtype)
print(f"X_train_mb.shape: {X_train_mb.shape}")
print(f"y_train_mb.shape: {y_train_mb.shape}")

net = Net(X_train_mb.shape[-1], 5, y_train_mb.shape[-1])

# %% Register hooks to the net

if 'hooks' in locals():
    for hook in hooks:
        hook.remove()

hooks=[]
hooks.append(
    net.lif1.register_forward_hook(partial(record_outputs, label='lif1')))
hooks.append(
    net.lif2.register_forward_hook(partial(record_outputs, label='lif2')))

# %% Check the only trainable parameters are from Linear layers

for name, param in net.named_parameters():
    print(f'Parameter: {name}, Requires Grad: {param.requires_grad}')

# %% Training

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, betas=(0.9, 0.999))

num_epochs = 100
loss_hist = []

wandb.watch(net, criterion, log="all", log_freq=1)
for epoch in range(num_epochs):
    iter_counter = 0

    # batch-wise training
    w_avg_hist = []
    for xi, yi in zip(X_train_mb, y_train_mb):
        optimizer.zero_grad()
        net.train()
        spk, mem = net(xi)

        #  compute the loss & sum over inner steps
        loss_val = sum([
           criterion(mem.squeeze(), yi)
           for step in range(net.inner_steps)])

        loss_hist.append(loss_val.item())
        w_avg_hist.append(net.fc2.weight.data.mean())

        # Gradient calculation + weight update
        loss_val.backward()
        optimizer.step()

        iter_counter +=1
        
        print(f"\nIteration: {iter_counter}; Epoch # {epoch}")
        print(f"Training loss: {loss_val}")
        print(f"fc2 weights: {net.fc2.weight.data.mean()}")
    wandb.log({'epoch': epoch, 'loss': loss_val})


# Visualize
fig, axs = plt.subplots(2, 1)
axs[0].plot(range(len(loss_hist)), loss_hist, label='Training Loss')
axs[0].legend()

axs[1].plot(w_avg_hist)
axs[1].set_title("FC2 layer Weights mean")
fig.tight_layout()

# %%

# Hooks remove
if 'hooks' in locals():
    for hook in hooks:
        hook.remove()
    del hooks

# %% Stack the recs

# mem_rec['lif1'] [=]
#       (n_minibatches * num_epochs * net.inner_steps, bs, lif1.output_size)
#       [=] (X_train_mb[0]*X_train_mb[1]*net.inner_steps, bs, lif1.output_size)
#
mem_rec['lif1'] = torch.stack(mem_rec['lif1']).detach()
mem_rec['lif2'] = torch.stack(mem_rec['lif2']).detach()
spk_rec['lif1'] = torch.stack(spk_rec['lif1']).detach()
spk_rec['lif2'] = torch.stack(spk_rec['lif2']).detach()

# See the mem and spk histories
for layer_name, mem in mem_rec.items():
    print(f"{layer_name} mem_rec: {mem_rec[layer_name].shape}")
    print(f"{layer_name} spk_rec: {spk_rec[layer_name].shape}")

# %% Visualize training

# Show the output along the training

var_idx = 2  # the lorenz system output variable index

# First portion of the training
range_to_plot = (
    slice(0, 100), # iterations
    slice(0, X_train_mb.shape[1]),  # bs
    var_idx
)

fig, axs = plt.subplots(3, 1)
axs[0].plot(X_train_mb[range_to_plot].flatten(), color='orange')
axs[0].set_title(f'Training Input, var #{var_idx}')
axs[1].plot(mem_rec['lif2'][range_to_plot].flatten().detach())
axs[1].axhline(net.lif2.threshold, linestyle='-.', color='grey')
axs[1].set_title(f'LIF2 neuron #{var_idx} Membrane Potential')
axs[2].plot(spk_rec['lif2'][range_to_plot].flatten().detach())
# axs[2].eventplot(torch.argwhere(spk_rec['lif1'][range_to_plot].flatten()).numpy().flatten())
axs[2].set_title(f'LIF2 neuron #{var_idx} Spikes')
fig.tight_layout()
fig.savefig(EXP_DATA_DIR/"LIF-net-Lorenz_first")

# Last portion of the training
range_to_plot = (
    slice(-80, X_train_mb.shape[0]), # iterations
    slice(0, X_train_mb.shape[1]),  # bs
    var_idx
)
fig, axs = plt.subplots(3, 1)
axs[0].plot(X_train_mb[range_to_plot].flatten(), color='orange')
axs[0].set_title(f'Training Input, Lorenz var #{var_idx}')
axs[1].plot(mem_rec['lif2'][range_to_plot].flatten().detach())
axs[1].axhline(net.lif2.threshold, linestyle='-.', color='grey')
axs[1].set_title(f'LIF2 neuron #{var_idx} Membrane Potential')
axs[2].plot(spk_rec['lif2'][range_to_plot].flatten().detach())
# axs[2].eventplot(torch.argwhere(spk_rec['lif1'][range_to_plot].flatten()).numpy().flatten())
axs[2].set_title(f'LIF2 neuron #{var_idx} Spikes')
fig.tight_layout()
fig.savefig(EXP_DATA_DIR/"LIF-net-Lorenz_last")


# %%

# (iterations, inner_steps, bs, layer_output_size)

# mem_rec_lif2 = torch.stack(
#     [mem_rec[i]['lif2'] for i in range(len(mem_rec))])
# spk_rec_lif2 = torch.stack(
#     [spk_rec[i]['lif2'] for i in range(len(spk_rec))])

# %% Plot: What the last layer see? -old notes, but useful-
#
# Here we expand the input, because the model has an internal dynamics,
#   during which the neurons evolve.
# So, the expanded signal will have a repetition of values for each net
#   inner step.
# e.g. signal:[1,2,3], net.inner_steps=3; => expanded: [1,1,1, 2,2,2, 3,3,3]

# n_i = 'lif2'
# range_to_plot = slice(500, 650)
# signal = X_train[:, 2].unsqueeze(1).expand(-1, net.inner_steps).reshape(-1)
# src09.utils.plot_cur_mem_spk(
#     # X_train[:, 2],
#     signal[range_to_plot], # <- input to the network
#     mem_rec_lif2.view(-1, 1).detach()[range_to_plot],  # (inner_steps, bs, 1)
#     spk_rec_lif2.view(-1, 1).detach()[range_to_plot],
#     thr_line=1,
#     ylim_input=(-20, 80),
#     ylim_mempot=(-1, 1),
#     x_lim=(0, range_to_plot.stop - range_to_plot.start),
#     title="LIF Neuron Model")
# %%
