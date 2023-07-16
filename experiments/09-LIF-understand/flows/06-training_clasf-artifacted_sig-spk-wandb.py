""" Training a Network on a signal with artifacts

Classification: Y = 0/1, where (Y|X) = 0 and (Y|X+art) = 1

1. A signal is generated along with artifacts.
2. A SNN is created as a Input + LIF + LIF.
    1. Each LIF has not trainable units
    2. The network has trainable parameters (Linear layers)
3. The input examples to the network are mini batches of dims (batch_size, 1),
    while the output labels are (batch_size, 1).



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
import wandb


sys.path += ['.', '../', '../..', '../../../', 'experiments', 'experiments/09-LIF-understand']
from experimentkit_in.funx import pickle_save_dict, pickle_load
from experimentkit_in.generators.time_series import gen_simple_sin, \
    gen_random_artifacts, add_artifacts_to_signal
import src09

ROOT = Path('../../../')
# ROOT = Path('.')
DATA_DIR = ROOT/'data'
EXP_DIR = ROOT/'experiments/09-LIF-understand'
EXP_DATA_DIR = EXP_DIR/'data'
EXP_REPORT_DIR = EXP_DIR/'report'

DTYPE = torch.float

wandb.login()



sweep_configuration = {
    'method': 'random',
    'metric': 
    {
        'goal': 'minimize', 
        'name': 'val_loss'
    },
    'parameters': 
    {
        'num_epochs': {'values': [4, 10, 20]},
        'inner_steps': {'values': [1, 2, 5, 10, 20, 30]},
        'beta1': {'values': [0.1, 0.5, 1, 1.5, 5]},
        'beta2': {'values': [0.1, 0.5, 1, 1.5, 5]},
        'hidden_size': {'values': [1, 3, 5, 10, 20]},
     }
}

# 3: Start the sweep
sweep_id = wandb.sweep(
    sweep=sweep_configuration,
    project='sw|snnTorch-Artifacted|spk-1'
    )





assert EXP_DIR.exists(), \
    f"CWD: {Path.cwd()}\nROOT: {ROOT.absolute()}"



# %% Gen signal with random artifacts


signal_length = 10_000
num_artifacts = 40
min_artifact_length = 5
max_artifact_length = 50
min_artifact_frequency = 1
max_artifact_frequency = 10

main_signal = gen_simple_sin(signal_length, frequency=5)

artifacts = gen_random_artifacts(
    num_artifacts, min_artifact_length, max_artifact_length,
    min_artifact_frequency, max_artifact_frequency)

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

X = torch.tensor(dirty_signal).unsqueeze(1)
y = torch.tensor(labels).unsqueeze(1)

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



# %% Define Network

class Net(nn.Module):
    def __init__(
            self,
            input_size, hidden_size, output_size,
            inner_steps = 10,
            beta1=0.5, beta2=2):
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


bs = 20  # batch size
X_train_mb = X_train[:X_train.shape[0]//bs * bs].reshape(
    X_train.shape[0]//bs, bs, X_train.shape[-1]).to(dtype)
y_train_mb = y_train[:y_train.shape[0]//bs * bs].reshape(
    y_train.shape[0]//bs, bs, y_train.shape[-1]).to(dtype)
print(f"X_train_mb.shape: {X_train_mb.shape}")
print(f"y_train_mb.shape: {y_train_mb.shape}")

# %%
def wandb_main():
    wandb.init(project='sweep-snnTorch-Artifacted', entity='gianfa')
    print(f"wandb.config: {wandb.config}")
    net = Net(
        input_size=X_train_mb.shape[-1],
        hidden_size=wandb.config.hidden_size,
        output_size=y_train_mb.shape[-1],
        inner_steps=wandb.config.inner_steps,
        beta1=wandb.config.beta1,
        beta2=wandb.config.beta2,
        )

    # Prepare the containers for recording mem pot and spike histories
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


    # Training
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, betas=(0.9, 0.999))

    num_epochs = wandb.config.num_epochs
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
                criterion(
                    spk.to(DTYPE), yi.to(DTYPE)
                )
                    for step in range(net.inner_steps)
            ])

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

    # TOWANDB
    # Stack the recs
    mem_rec['lif1'] = torch.stack(mem_rec['lif1']).detach()
    mem_rec['lif2'] = torch.stack(mem_rec['lif2']).detach()
    spk_rec['lif1'] = torch.stack(spk_rec['lif1']).detach()
    spk_rec['lif2'] = torch.stack(spk_rec['lif2']).detach()

    # See the mem and spk histories
    for layer_name, memi in mem_rec.items():
        print(f"{layer_name} mem_rec: {mem_rec[layer_name].shape}")
        print(f"{layer_name} spk_rec: {spk_rec[layer_name].shape}")

    # Visualize training
    print(f"X_train_mb.shape: {X_train_mb.shape}")
    # var_idx = 2
    # range_to_plot = (
    #     slice(0, X_train_mb.shape[0]), # iterations
    #     slice(0, X_train_mb.shape[1]),  # bs
    #     var_idx
    # )

    # fig, axs = plt.subplots(3, 1)
    # axs[0].plot(X_train_mb[range_to_plot].flatten(), color='orange')
    # axs[0].set_title(f'Training Input, var #{var_idx}')
    # axs[1].plot(mem_rec['lif2'][range_to_plot].flatten().detach())
    # axs[1].axhline(net.lif2.threshold, linestyle='-.', color='grey')
    # axs[1].set_title(f'LIF2 neuron #{var_idx} Membrane Potential')
    # axs[2].plot(spk_rec['lif2'][range_to_plot].flatten().detach())
    # # axs[2].eventplot(torch.argwhere(spk_rec['lif1'][range_to_plot].flatten()).numpy().flatten())
    # axs[2].set_title(f'LIF2 neuron #{var_idx} Spikes')

    # /TOWANDB

    # table = wandb.Table(
    #     data=list(enumerate(X_train_mb[range_to_plot].flatten())),
    #     columns=["dt", "I_in"])
    # wandb.log(
    #     {"my_custom_plot_id" : wandb.plot.line(table, "dt", "I_in",
    #         title="Input current")})
    
    # table = wandb.Table(
    #     data=list(enumerate(mem_rec[range_to_plot].flatten())),
    #     columns=["dt", "mem potential"])
    # wandb.log(
    #     {"my_custom_plot_id" : wandb.plot.line(table, "dt", "mem potential",
    #         title=f"LIF2 neuron #{var_idx} Membrane Potential")})


# %%

wandb.agent(sweep_id, function=wandb_main, count=10)

print(wandb.config)



# %%

# (iterations, inner_steps, bs, layer_output_size)

# mem_rec_lif2 = torch.stack(
#     [mem_rec_hist[i]['lif2'] for i in range(len(mem_rec_hist))])
# spk_rec_lif2 = torch.stack(
#     [spk_rec_hist[i]['lif2'] for i in range(len(spk_rec_hist))])

# %% Plot: What the last layer see?
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
