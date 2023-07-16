""" Training a Network on a signal with artifacts

Classification on sliding window: Y = 0/1, where (Y|X) = 0 and (Y|X+art) = 1

1. A signal is generated along with artifacts.
2. A SNN is created as a Input + LIF + LIF.
    1. Each LIF has not trainable units
    2. The network has trainable parameters (Linear layers)
3. The input examples to the network are mini batches of dims (batch_size, 1),
    while the output labels are (batch_size, 1).
    1. all input_examples are generate by a window sliding over the generated 
        signal with the artifact.



NOTES
-----
Weights are dead. Maybe the task vs the loss need improvement

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
from experimentkit_in.generators.time_series import gen_simple_sin, \
    gen_random_artifacts, add_artifacts_to_signal, ts_gen_signal_shifts
import src09

import wandb
wandb.login()
wandb.init(project='snnTorch-Artifacted', entity='gianfa')

ROOT = Path('../../../')
DATA_DIR = ROOT/'data'
EXP_DIR = ROOT/'experiments/09-LIF-understand'
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'

assert EXP_DIR.exists(), \
    f"CWD: {Path.cwd()}\nROOT: {ROOT.absolute()}"

DTYPE = torch.float

# %% Gen signal with random artifacts


signal_length = 10_000
num_artifacts = 40
min_artifact_length = 5
max_artifact_length = 50
min_artifact_frequency = 1
max_artifact_frequency = 10

main_signal = gen_simple_sin(signal_length, frequency=20)

artifacts = gen_random_artifacts(
    num_artifacts,
    min_artifact_length,
    max_artifact_length,
    min_artifact_frequency,
    max_artifact_frequency)

dirty_signal, labels = add_artifacts_to_signal(main_signal, artifacts)

# Plot
fig, axs = plt.subplots(3, 1)
axs[0].plot(main_signal)
axs[0].set_title('main_signal')
axs[1].plot(dirty_signal)
axs[1].set_title('dirty_signal')
axs[2].plot(labels)
axs[2].set_title('labels indices')
fig.tight_layout()

# %%

window_size = 400

shifted_sigs = torch.Tensor(
    ts_gen_signal_shifts(dirty_signal, window_size=window_size, dim=0))
"""Shifted signals
[=](n_signals, windows_size)
"""

shifted_labels = torch.Tensor(
    ts_gen_signal_shifts(labels, window_size=window_size, dim=0))
"""Shifted labels
[=](n_labels_sigs, windows_size)
"""

# plt.plot(shifted_sigs[20, :])
fig, ax = plt.subplots()
for i in range(100):
    ax.plot(shifted_sigs[i, :] + 0.2)
ax.set_title("Example of shifted signals")

print(
    f"X_shifted_sigs.shape: {shifted_sigs.shape}" +
    f"\nshifted_labels.shape: {shifted_labels.shape}")


X = torch.tensor(shifted_sigs).unsqueeze(1)  # [=] (signal_length, 1)
y = torch.tensor(shifted_labels).unsqueeze(1)  # [=] (signal_length, 1)

# %% Train/Test split

test_size = 0.2
valid_size = 0.15

X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=0.2, shuffle=False)

valid_size = 0.15 / (1 - test_size)
X_train, X_valid, y_train, y_valid = \
    train_test_split(X_train, y_train, test_size=0.2, shuffle=False)

# X_train [=] (n_fragments, fragment_len)
X_train = X_train.to(DTYPE)
X_valid = X_valid.to(DTYPE)
X_test = X_test.to(DTYPE)
y_train = y_train.to(DTYPE).unsqueeze(1)
y_valid = y_valid.to(DTYPE).unsqueeze(1)
y_test = y_test.to(DTYPE).unsqueeze(1)

print(
    X_train.shape, y_train.shape,
    X_valid.shape, y_valid.shape,
    X_test.shape, y_test.shape)


# %% Define the Network

class Net(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            beta1: float = 1.5,
            beta2: float = 5,
            inner_steps: int = 1
        ):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.lif1 = snn.Leaky(beta=beta1, threshold=15)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.lif2 = snn.Leaky(beta=beta2, threshold=10)
        self.inner_steps = inner_steps

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


bs = 30  # batch size

# X_train_mb [=] (batches, bs, input_size) = (batches, bs, fragment_len)
X_train_mb = X_train[:X_train.shape[0]//bs * bs].reshape(
    X_train.shape[0]//bs, bs, X_train.shape[-1]).to(dtype)

# y_train_mb [=] (batches, bs, output_size)
y_train_mb = y_train[:y_train.shape[0]//bs * bs].reshape(
    y_train.shape[0]//bs, bs, y_train.shape[-1]).to(dtype)

# Plot batches
# for i in range(10):
#     plt.plot(X_train_mb[i, :, :].flatten() + 0.2)

print(f"X_train_mb.shape: {X_train_mb.shape}")
print(f"y_train_mb.shape: {y_train_mb.shape}")

hidden_size = 10
net = Net(
    input_size=X_train_mb.shape[-1],
    hidden_size=hidden_size,
    output_size=y_train_mb.shape[-1],
    beta1=torch.linspace(0.05, 3, hidden_size),
    beta2=torch.linspace(2, 10, y_train_mb.shape[-1]),
    inner_steps=10,
    )



# %% Prepare the containers for recording mem pot and spike histories
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

# Register hooks to the net
if 'hooks' in locals():
    for hook in hooks:
        hook.remove()

hooks=[]
hooks.append(
    net.lif1.register_forward_hook(partial(record_outputs, label='lif1')))
hooks.append(
    net.lif2.register_forward_hook(partial(record_outputs, label='lif2')))

# %%
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-3)

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
        # spk_rec_out_last = torch.stack(
        #     spk_rec['lif2']).detach()[-net.inner_steps:]
        # loss_val = sum([
        #         criterion(spk_i.to(DTYPE), yi.to(DTYPE))
        #             for spk_i in spk_rec_out_last
        #     ])

        # #  Compute loss only on the output mem potentials
        # loss_val = criterion(
        #             mem.to(DTYPE), yi.to(DTYPE) * net.lif2.threshold.item()
        #         )

        # #  Compute loss over the last inner loop mem potentials
        # get the last `inner_steps` iterations
        inner_loop_spk = torch.stack(spk_rec['lif2'][-net.inner_steps:])
        # expand the label tensor consistently with spk_rec
        expanded_yi = yi.unsqueeze(0).expand(
            net.inner_steps,
            bs,
            window_size)
        # compute loss_val
        loss_val = criterion(
            inner_loop_spk.to(DTYPE),
            expanded_yi.to(DTYPE))

        loss_hist.append(loss_val.item())
        w_avg_hist.append(net.fc2.weight.data.mean())

        # Gradient calculation + weight update
        loss_val.backward()
        optimizer.step()

        iter_counter +=1
        
        print(f"\nIteration: {iter_counter}; Epoch # {epoch}")
        print(f"Training loss: {loss_val}")
        print(f"fc2 weights: {net.fc2.weight.data.mean()}")
    
    wandb.log({
            'epoch': epoch,
            'loss': loss_val,
            'weights_avg': net.fc2.weight.data.mean()
            })



# Plot Loss
fig, axs = plt.subplots(2, 1)
axs[0].plot(range(len(loss_hist)), loss_hist)
axs[0].set_title("Training Loss")
axs[0].legend()
axs[1].plot(w_avg_hist)
axs[1].set_title("FC2 layer Weights mean")
fig.tight_layout()

# Hooks remove
if 'hooks' in locals():
    for hook in hooks:
        hook.remove()
    del hooks

# %%

# (iterations * inner_steps, bs, layer_output_size)

# Stack the recs
mem_rec['lif1'] = torch.stack(mem_rec['lif1']).detach()
mem_rec['lif2'] = torch.stack(mem_rec['lif2']).detach()
spk_rec['lif1'] = torch.stack(spk_rec['lif1']).detach()
spk_rec['lif2'] = torch.stack(spk_rec['lif2']).detach()

# See the mem and spk histories
for layer_name, memi in mem_rec.items():
    print(f"{layer_name} mem_rec: {mem_rec[layer_name].shape}")
    print(f"{layer_name} spk_rec: {spk_rec[layer_name].shape}")

# %%

# Visualize training
print(f"X_train_mb.shape: {X_train_mb.shape}")
var_idx = 2
range_to_plot = (
    slice(0, X_train_mb.shape[0]), # iterations
    slice(0, X_train_mb.shape[1]), # bs
    var_idx
)

fig, axs = plt.subplots(3, 1)
axs[0].plot(X_train_mb[range_to_plot].flatten(), color='orange')
axs[0].set_title(f'Training Input, var #{var_idx}')
axs[1].plot(mem_rec['lif2'][range_to_plot].flatten().detach())
axs[1].axhline(net.lif2.threshold, linestyle='-.', color='grey')
axs[1].set_title(f'LIF2 neuron #{var_idx} Membrane Potential')
# axs[2].plot(spk_rec['lif2'][range_to_plot].flatten().detach())
spks_sig = spk_rec['lif2'][range_to_plot].flatten()
axs[2].eventplot(torch.argwhere(spks_sig).numpy().flatten())
axs[2].set_xlim(0, len(spks_sig)) 
axs[1].set_xlim(0, len(spks_sig)) 
axs[0].set_xlim(0, len(spks_sig)) 
#     range_to_plot[0].start,
#     (range_to_plot[0].stop - range_to_plot[0].start) * )
axs[2].set_title(f'LIF2 neuron #{var_idx} Spikes')
fig.tight_layout()

# %% Plot: What the last layer see?
# Here we expand the input, because the model has an internal dynamics,
#   during which the neurons evolve.
# So, the expanded signal will have a repetition of values for each net
#   inner step.
# e.g. signal:[1,2,3], net.inner_steps=3; => expanded: [1,1,1, 2,2,2, 3,3,3]

lif_idx = 'lif2'
range_to_plot = slice(500, 1300)


# X_train_mb [=] (n_batches, bs, fragment_len)
# expanded_X [=] (n_batches * inner_steps, bs, fragment_len)
expanded_X = \
    X_train_mb.unsqueeze(1).expand(
        X_train_mb.shape[0],
        net.inner_steps,
        X_train_mb.shape[1],
        X_train_mb.shape[2],
    ).reshape(
        X_train_mb.shape[0] * net.inner_steps,
        X_train_mb.shape[1],
        X_train_mb.shape[2],
    )

expanded_y = \
    y_train_mb.unsqueeze(1).expand(
        y_train_mb.shape[0],
        net.inner_steps,
        y_train_mb.shape[1],
        y_train_mb.shape[2],
    ).reshape(
        y_train_mb.shape[0] * net.inner_steps,
        y_train_mb.shape[1],
        y_train_mb.shape[2],
    )

n_id = 0
src09.utils.plot_cur_mem_spk(
    # X_train[:, 2],
    expanded_X.flatten()[range_to_plot], # <- input to the network
    mem_rec[lif_idx] \
        .view(-1, mem_rec[lif_idx][n_id].shape[-1]) \
        [:, 0] \
        .detach()[range_to_plot],  # [=] (steps * inner_steps, n_id)
    spk_rec[lif_idx] \
        .view(-1, mem_rec[lif_idx][n_id].shape[-1]) \
        [:, 0] \
        .detach()[range_to_plot],  # [=] (steps * inner_steps, n_id)
    thr_line=net.lif2.threshold,
    ylim_input=(-2, 10),
    ylim_mempot=(-15, 12),
    x_lim=(0, range_to_plot.stop - range_to_plot.start),
    title="LIF Neuron Model")

# %% Visualization [working]

n_id = 0
range_to_plot = slice(500, 1300)

src09.utils.plot_cur_mem_spk(

    expanded_X.flatten()[range_to_plot], # <- input to the network
    
    mem_rec[lif_idx].view(-1, window_size)[range_to_plot, n_id].detach(),
    
    spk_rec[lif_idx].view(-1, window_size)[range_to_plot, n_id].detach(),
    
    thr_line=net.lif2.threshold,
    ylim_input=(-2, 10),
    ylim_mempot=(-15, 12),
    x_lim=(0, range_to_plot.stop - range_to_plot.start),
    title="LIF Neuron Model")



# %% Many neurons plot: membrane potential

neuron_idxs = list(range(10))
range_to_plot = slice(500, 1300)

fig, axs = plt.subplots(len(neuron_idxs) + 1)
axs[0].plot(expanded_X.flatten()[range_to_plot], color='orange')

for ax, n_id in zip(axs[1:].ravel(), neuron_idxs):
    ax.plot(
        mem_rec[lif_idx].view(-1, window_size)[range_to_plot, n_id].detach())
    ax.set_ylabel(f"#{n_id}")
    if n_id < len(neuron_idxs) -  1:
        ax.set_xticklabels([])

fig.suptitle("Neurons membrane potential")
fig.tight_layout()

# %% Many neurons plot: spikes

neuron_idxs = list(range(10))
range_to_plot = slice(500, 1300)

fig, axs = plt.subplots(len(neuron_idxs) + 1)
axs[0].scatter(
    spk_input_idxs,
    torch.ones(len(spk_input_idxs)) * 1,
    color='orange',
    marker="|"
)
mem_rec[lif_idx].view(-1, window_size)[range_to_plot, n_id].detach()

for ax, n_id in zip(axs[1:].ravel(), neuron_idxs):
    ax.plot(
        spk_rec[lif_idx].view(-1, window_size)[range_to_plot, n_id].detach())
    ax.set_ylabel(f"#{n_id}")
    if n_id < len(neuron_idxs) -  1:
        ax.set_xticklabels([])

fig.suptitle("Neurons membrane potential")
fig.tight_layout()

# %% %% Plot: What the last layer see?
# Here we expand the input, because the model has an internal dynamics,
#   during which the neurons evolve.
# So, the expanded signal will have a repetition of values for each net
#   inner step.
# e.g. signal:[1,2,3], net.inner_steps=3; => expanded: [1,1,1, 2,2,2, 3,3,3]

lif_idx = 'lif2'
var_idx = 2
range_to_plot = (
    slice(0, X_train_mb.shape[0]), # iterations
    slice(0, X_train_mb.shape[1]), # bs
    var_idx
)

# X_train_mb [=] (n_batches, bs, fragment_len)
# expanded_X [=] (n_batches * inner_steps, bs, fragment_len)
expanded_X = \
    X_train_mb.unsqueeze(1).expand(
        X_train_mb.shape[0],
        net.inner_steps,
        X_train_mb.shape[1],
        X_train_mb.shape[2],
    ).reshape(
        X_train_mb.shape[0] * net.inner_steps,
        X_train_mb.shape[1],
        X_train_mb.shape[2],
    )

expanded_y = \
    y_train_mb.unsqueeze(1).expand(
        y_train_mb.shape[0],
        net.inner_steps,
        y_train_mb.shape[1],
        y_train_mb.shape[2],
    ).reshape(
        y_train_mb.shape[0] * net.inner_steps,
        y_train_mb.shape[1],
        y_train_mb.shape[2],
    )

# Visualize training
print(f"X_train_mb.shape: {X_train_mb.shape}")


fig, axs = plt.subplots(4, 1)
axs[0].plot(expanded_X[range_to_plot].flatten(), color='orange')
axs[0].set_title(f'Training Input, var #{var_idx}')
axs[1].plot(mem_rec[lif_idx][range_to_plot].flatten().detach())
axs[1].axhline(net.lif2.threshold, linestyle='-.', color='grey')
axs[1].set_title(f'{lif_idx} neuron #{var_idx} Membrane Potential')

# Label spike
axs[2].eventplot(torch.argwhere(expanded_y[range_to_plot]).numpy().flatten())
axs[2].set_title(f'yi spikes, neuron #{var_idx}')

# LF spike
spks_sig = spk_rec[lif_idx][range_to_plot]
axs[3].eventplot(torch.argwhere(spks_sig).numpy().flatten())

axs[3].set_xlim(0, len(spks_sig.flatten())) 
axs[2].set_xlim(0, len(spks_sig.flatten())) 
axs[1].set_xlim(0, len(spks_sig.flatten())) 
axs[0].set_xlim(0, len(spks_sig.flatten())) 
#     range_to_plot[0].start,
#     (range_to_plot[0].stop - range_to_plot[0].start) * )
axs[3].set_title(f'LIF2 neuron #{var_idx} Spikes')
fig.tight_layout()
# %%
