""" Simple LSM, not learning




        ┌─────|lif_lif|──┐
                ߇        │
I ----|lif_I|---------> LIF_1 ---|lif_ro|---> readout -->

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
    torch.ones(110) * 0.8,
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


readout = snn.Lapicque(
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


def thresholded_relu(x, threshold):
    return torch.where(x > threshold, x, 0)


# Define the connections between input and LIF
# This is intended like a static synapses adjacency matrix
conn_lif_I = torch.tensor([.2, .5]).reshape(2, 1)

# Define the connection matrix between LIF and LIF
# This is intended like a static synapses adjacency matrix
conn_lif_lif = torch.tensor([
    [0, .5],
    [.5, 0]])

# Define the connections between LIF and readout
# This is intended like a static synapses adjacency matrix
conn_readout_lif = torch.tensor([1., .7]).reshape(2, 1)

# ---- Simulate ----

# History variables
mem_filter = [torch.zeros(2, 1)]
spk_filter = [torch.zeros(2, 1)]
activ_filter = [torch.zeros(2, 1)]

mem_readout = [torch.zeros(1, 1)]
spk_readout = [torch.zeros(1, 1)]
activ_readout = [torch.zeros(1, 1)]

input = []

# %%
# neuron simulation
for t in range(1, len(I) -1):
    # Current input: Hadamard I . conn_lif_I
    cur_input = I[t] * torch.ones(2, 1)
    
    # Recurrent input
    rec_input = torch.matmul(conn_lif_lif, activ_filter[t-1])

    ### Input ###
    input_t = torch.mul(
        cur_input + rec_input,
        conn_lif_I).reshape(2, 1)

    ### Filter ###
    # lif1 needs 1D args
    # spk_t, mem_t = lif1( I[t] + lif_lif_input[t-1], mem[t-1])
    spk_filter_t, mem_filter_t = lif1(
        input_t.squeeze(), mem_filter[t-1].squeeze())
    # make them columnar
    mem_filter_t = mem_filter_t.reshape(len(mem_filter_t), 1)
    spk_filter_t = spk_filter_t.reshape(len(mem_filter_t), 1)

    activ_filter_t = thresholded_relu(1 * mem_filter_t, lif1.threshold)

    ### Readout ###
    readout_input = torch.matmul(
        conn_readout_lif.T,
        activ_filter_t).reshape(1, 1)
    spk_readout_t, mem_readout_t = readout(
        readout_input, mem_readout[t-1].squeeze())
    mem_readout_t = mem_readout_t.reshape(1, 1)
    spk_readout_t = spk_readout_t.reshape(1, 1)

    activ_readout_t = thresholded_relu(1 * mem_readout_t, lif1.threshold)

    # store current state
    input.append(input_t)
    mem_filter.append(mem_filter_t)
    spk_filter.append(spk_filter_t)
    activ_filter.append(activ_filter_t)

    mem_readout.append(mem_readout_t)
    spk_readout.append(spk_readout_t)
    activ_readout.append(activ_readout_t)

assert abs(I.shape[0] - len(input)) <= 2
assert (
    len(input) + 1 == len(mem_filter) == len(spk_filter) == len(activ_filter))

# convert lists to tensors
input = torch.stack(input)
mem_filter = torch.stack(mem_filter)
spk_filter = torch.stack(spk_filter)
activ_filter = torch.stack(activ_filter)
mem_readout = torch.stack(mem_readout)
spk_readout = torch.stack(spk_readout)
activ_readout = torch.stack(activ_readout)

# %%
# ---- Visualize ----

axs = []
added_plots = 3
tot_plots = added_plots + mem_filter.shape[1] + mem_readout.shape[1]
fig, axs = plt.subplots(tot_plots, 1, sharex=True, figsize=(7, 12))

axs[0].plot(I, c="red")
axs[0].set_ylabel(f"$I$", rotation=0)
axs[0].grid()

axs[1].plot(input.squeeze()[:, 1], c="orange")
axs[1].set_ylabel(r"$inp_{I0}$", rotation=0)
axs[1].grid()

axs[2].plot(input.squeeze()[:, 0], c="orange")
axs[2].set_ylabel(r"$inp_{I1}$", rotation=0)
axs[2].grid()

for i, ax in enumerate(
    axs.ravel()[added_plots:added_plots + mem_filter.shape[1]]):
    if i == 0:
        ax.set_title(f"Filter")
    ax.plot(mem_filter.squeeze()[:, i])
    ax.set_ylabel(f"$V_n{i}$", rotation=0)
    ax.grid()
    ax.set_xlabel(r"$t_{i}$")

for i, ax in enumerate(axs.ravel()[added_plots + mem_filter.shape[1]:]):
    if i == 0:
        ax.set_title(f"Readout")
    ax.plot(mem_readout[:, i])
    ax.set_ylabel(r"$V_{out}$", rotation=0)
    ax.grid()
    ax.set_xlabel(r"$t_{i}$")

fig.tight_layout()

# %%
