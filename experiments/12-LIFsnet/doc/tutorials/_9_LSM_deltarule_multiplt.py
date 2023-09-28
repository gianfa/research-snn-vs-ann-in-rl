""" Simple LSM - learning: fitting [WIP]

Higher dimension

Fed by a periodic signal

        ┌─|lif_lif|──┐
        ߇            │
I ----|lif_I|----> LIF_1 ---|lif_ro|---> readout -->


We start with a binary fitting.
X: 0/0.3
y: 0/1; 1 corresponds to 0.3

Notes:
1. Set initial weigths > 0; e.g. 1e-1
2. Don't go with a neuronal input < 0.
3. Consider a decoder between y_pred and labels

"""
# %% Neurons pair dynamics

import snntorch as snn
import torch
import matplotlib.pyplot as plt

n = 1000  # signal length (simulation step)
duration = 3  # duration of a signal peak

lr = 0.9  # learning rate
error_scope = 4  # n of steps to average over, in order to compute the error

# ---- Define the neuron and the input current -----

# # Examples (X) definition
I = torch.zeros(n) + 0.35  # baseline
I_pos_idxs = torch.randperm(len(I))[:int(0.08 * n)]
for idx in I_pos_idxs:
    I[idx:idx+duration] = 0.5

# # Labels (y) definition
y = I.clone()
for idx in I_pos_idxs:
    y[idx:idx+duration] = 1

fig, axs = plt.subplots(2, 1)
axs[0].plot(torch.arange(len(I)), I)
axs[0].set_title('I')
axs[0].set_ylabel('amplitude')
axs[0].set_xlabel('t')
axs[0].grid()

axs[1].plot(torch.arange(len(y)), y)
axs[1].set_title('Labels (y)')
axs[1].set_ylabel('')
axs[1].set_xlabel('t')
axs[1].set_yticks([-1, 0, 1])
axs[1].grid()
fig.tight_layout()

# %%

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
conn_lif_I = torch.tensor(
    [0.8, 0.2, 0.3, 0.5, 0.6]).reshape(5, 1)

# Define the connection matrix between LIF and LIF
# This is intended like a static synapses adjacency matrix
conn_lif_lif = torch.tensor([
    [0, -0.5, .9 , .6, .2],
    [0,  0, 0 , 0, .2],
    [.2, -.4, 0 , 0, .2],
    [0, .1, 0 , 0, .2],
    [1., .3, 0 , -.2,  0],
    ])

# Define the connections between LIF and readout
# This is intended like a static synapses adjacency matrix
conn_readout_lif = torch.tensor(
    [0.8, 0.2, -0.3, 0.9, 0.6]).reshape(5, 1)

# ---- Simulate ----

# History variables
mem_filter = [torch.zeros(5, 1)]
spk_filter = [torch.zeros(5, 1)]
activ_filter = [torch.zeros(5, 1) + 1e-2]

mem_readout = [torch.zeros(1, 1) + 1e-2]
spk_readout = [torch.zeros(1, 1)]
activ_readout = [torch.zeros(1, 1) + 1e-2]
conn_readout_lif_coll = [conn_readout_lif]

input = []
rec_input = []
readout_input = []

# %%
# neuron simulation
error_sum = 0
mse = []
for t in range(1, len(I) -1):
    # Current input: Hadamard I . conn_lif_I
    cur_input = I[t] * torch.ones(5, 1)
    
    # Recurrent input
    rec_input_t = torch.matmul(conn_lif_lif, activ_filter[t-1])

    ### Input ###
    input_t = torch.mul(
        cur_input + rec_input_t,
        conn_lif_I).reshape(5, 1)

    ### Filter ###
    # lif1 needs 1D args
    spk_filter_t, mem_filter_t = lif1(
        input_t.squeeze(), mem_filter[t-1].squeeze())
    # make them columnar
    mem_filter_t = mem_filter_t.reshape(len(mem_filter_t), 1)
    spk_filter_t = spk_filter_t.reshape(len(mem_filter_t), 1)

    activ_filter_t = thresholded_relu(1 * mem_filter_t, lif1.threshold)

    ### Readout ###
    readout_input_t = torch.matmul(
        conn_readout_lif.T,
        activ_filter_t).reshape(1, 1)
    spk_readout_t, mem_readout_t = readout(
        readout_input_t, mem_readout[t-1].squeeze())
    mem_readout_t = mem_readout_t.reshape(1, 1)
    spk_readout_t = spk_readout_t.reshape(1, 1)

    activ_readout_t = thresholded_relu(
        1 * mem_readout_t, readout.threshold)

    ### Learning ###
    # compute error
    pred_y = activ_readout_t * 1/readout.threshold  # activation * decoder
    error = y[t] - pred_y
    error_sum += error

    # update weigths
    # opt1. step-wise
    # conn_readout_lif += (lr * error * activ_filter_t)
    # opt2. scope-wise
    if t % error_scope == 0:
        conn_readout_lif += (lr * error.mean() * activ_filter_t)


    if t % error_scope == 0:
        mse_t = error_sum**2 / error_scope
        mse.append(mse_t.item())
        error_sum = 0
    

    # store current state
    input.append(input_t)
    rec_input.append(rec_input_t)
    mem_filter.append(mem_filter_t)
    spk_filter.append(spk_filter_t)
    activ_filter.append(activ_filter_t)

    mem_readout.append(mem_readout_t)
    spk_readout.append(spk_readout_t)
    activ_readout.append(activ_readout_t)
    readout_input.append(readout_input_t)

    conn_readout_lif_coll.append(conn_readout_lif.clone())

assert abs(I.shape[0] - len(input)) <= 2
assert (
    len(input) + 1 == len(mem_filter) == len(spk_filter) == len(activ_filter))

# convert lists to tensors
input = torch.stack(input)
rec_input = torch.stack(rec_input)
mem_filter = torch.stack(mem_filter)
spk_filter = torch.stack(spk_filter)
activ_filter = torch.stack(activ_filter)
mem_readout = torch.stack(mem_readout)
spk_readout = torch.stack(spk_readout)
readout_input = torch.stack(readout_input)
activ_readout = torch.stack(activ_readout)
conn_readout_lif_coll = torch.stack(conn_readout_lif_coll)
mse = torch.tensor(mse)

# Plot MSE
fig, ax = plt.subplots()
ax.plot(mse)
ax.set_title('MSE')
ax.set_xlabel('t / error_scope')
ax.axhline(0, linewidth=0.5)
ax.grid()

# Readout temporary plot
fig, ax = plt.subplots()
ax.plot(conn_readout_lif_coll[:, 0])
ax.set_title('conn_readout_lif_coll - 0')

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
