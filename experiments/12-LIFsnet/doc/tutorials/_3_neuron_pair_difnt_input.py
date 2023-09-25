""" Connected Neurons pair dynamics

Pay attention to the y axes of the final plot

$ python -m pytest test_3_neuron_pair_difnt_inputcopy.py -s --pdb
"""
# %% Neurons pair dynamics

import snntorch as snn
import torch
import matplotlib.pyplot as plt

# ---- Define the neuron and the input current -----

# def test_1():
# Small step current input
I = torch.cat(
    (torch.zeros(20),
    torch.ones(110) * 0.2,
    torch.zeros(80)), dim=0)

# Define the neuron
lif1 = snn.Lapicque(
    beta=False,
    R=5.1,
    C=5e-3,
    time_step=1e-3,
    threshold=0.5,
    spike_grad=None,  # surrogate gradient function
    init_hidden=False,
    inhibition=False,  # If `True`, suppresses all spiking other than the neuron with the highest state
    learn_beta=False,  # enable learnable beta
    learn_threshold=False,  # enable learnable threshold
    reset_mechanism="zero",  # {'subtract', zero', 'none'}
    state_quant=False,  # If `True` as well as `init_hidden=True`, states are returned
    output=False,
)

# Define the connections between input and LIF
# This is intended like a static synapses adjacency matrix
conn_lif_I = torch.tensor([.2, .05]).reshape(2, 1)

# ---- Simulate ----

# History variables
mem = [torch.zeros(2, 1)]  # here we impose the number of neurons
spk = [torch.zeros(2, 1)]
input = []

# %%
# neuron simulation
for t in range(1, len(I) -1):
    # Hadamard I . conn_lif_I
    input_t = torch.mul(
        I[t] * torch.ones(2, 1),
        conn_lif_I).reshape(2, 1)

    # lif1 needs 1D args
    # spk_t, mem_t = lif1( I[t] + lif_lif_input[t-1], mem[t-1])
    spk_t, mem_t = lif1( input_t.squeeze(), mem[t-1].squeeze())
    mem_t = mem_t.reshape(len(mem_t), 1)  # make it columnar
    spk_t = spk_t.reshape(len(mem_t), 1)  # make it columnar

    # store current state
    mem.append(mem_t)
    spk.append(spk_t)
    input.append(input_t)

# convert lists to tensors
mem = torch.stack(mem)
spk = torch.stack(spk)
input = torch.stack(input)

# ---- Visualize ----

axs = []
fig, axs = plt.subplots(mem.shape[1] + 2, 1, sharex=True)
fig.suptitle("Mem Pot")

axs[0].plot(input.squeeze()[:, 0], c="orange")
axs[0].set_ylabel(f"$I_{1}$", rotation=0)
axs[0].grid()

axs[1].plot(input.squeeze()[:, 1], c="orange")
axs[1].set_ylabel(f"$I_{2}$", rotation=0)
axs[1].grid()

for i, ax in enumerate(axs.ravel()[2:]):
    ax.plot(mem.squeeze()[:, i])
    ax.set_ylabel(f"$V_n{i}$", rotation=0)
    ax.grid()

# %%
