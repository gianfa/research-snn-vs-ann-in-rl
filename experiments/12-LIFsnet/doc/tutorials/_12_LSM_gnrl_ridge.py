""" Simple LSM - learning: fitting

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
4. Consider a I baseline so that the base-activity of the neurons is sustained.
5. Clip the activation shot (in thresholded relu). Otherwise it will shoot randomly.

Looking at the Readout plot with peaks red lines, it starts to be meaningful what we get.
Interesting is playing with I_baseline, peaks duration, error_scope.


"""
# %% Neurons pair dynamics

import snntorch as snn
import torch
import matplotlib.pyplot as plt

n = 1000  # signal length (simulation step)
duration = 1  # duration of a signal peak

# lr = 0.4  # learning rate
error_scope = 10  # n of steps to average over, in order to compute the error


# ---- Define the neuron and the input current -----

# # Examples (X) definition
I_baseline = 2
I_peak = 6
I = torch.zeros(n) + I_baseline  # baseline
I_pos_peaks_idxs = torch.randperm(len(I))[:int(0.08 * n)]
for idx in I_pos_peaks_idxs:
    I[idx:idx+duration] = I_peak

# # Labels (y) definition
y = torch.zeros_like(I)
y[torch.argwhere(I == I_peak)] = 1

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

# Define the reservoir
lif1_size = 10
lif1_inhib_frac = 0.2  # inhibitory fraction
lif1_sparsity = 0  # sparsity of the lif-lif connections 
lif1_has_autapsys = False
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

readout_size = 1
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


def thresholded_relu(x, threshold, clip_value=None):
    clip_value_ = clip_value or threshold * 1.2
    return torch.where(x > threshold, x.clip(max=clip_value_), 0)


lif_I_sparsity = 0  # sparsity of the lif-I connections
# Define the connections between input and LIF
# This is intended like a static synapses adjacency matrix
conn_lif_I = torch.rand(lif1_size).reshape(lif1_size, 1)
# -sparsity-
if lif1_sparsity > 0:
    lif_I_dense_idxs = torch.argwhere(conn_lif_I)
    lif_I_tooff_idxs = lif_I_dense_idxs[torch.randperm(
        len(lif_I_dense_idxs))[:int(lif1_size * lif_I_sparsity)]]
    conn_lif_I[lif_I_tooff_idxs[:, 0], lif_I_tooff_idxs[:, 1]] = 0


# Define the connection matrix between LIF and LIF
# This is intended like a static synapses adjacency matrix
# -inhibitors-
inhib_idxs = torch.randperm(lif1_size)[:int(lif1_size * lif1_inhib_frac)]
conn_lif_lif = torch.rand(lif1_size**2).reshape(lif1_size, lif1_size)
conn_lif_lif[inhib_idxs] = conn_lif_lif[inhib_idxs] * -1
# -autapsys-
if not lif1_has_autapsys:
    conn_lif_lif[torch.arange(lif1_size), torch.arange(lif1_size)] = 0
# -sparsity-
if lif1_sparsity > 0:
    lif1_dense_idxs = torch.argwhere(conn_lif_lif)
    lif1_tooff_idxs = lif1_dense_idxs[torch.randperm(
        len(lif1_dense_idxs))[:int(lif1_size**2 * lif1_sparsity)]]
    conn_lif_lif[lif1_tooff_idxs[:, 0], lif1_tooff_idxs[:, 1]] = 0
    # len(torch.argwhere(conn_lif_lif))/(lif1_size*lif1_size)

# Define the connections between LIF and readout
# This is intended like a static synapses adjacency matrix
conn_readout_lif = torch.rand(
    lif1_size * readout_size).reshape(lif1_size, readout_size)


# ---- Simulate ----

# History variables
mem_filter = [torch.zeros(lif1_size, 1)]
spk_filter = [torch.zeros(lif1_size, 1)]
activ_filter = [torch.zeros(lif1_size, 1) + 1e-2]

mem_readout = [torch.zeros(readout_size, 1) + 1e-2]
spk_readout = [torch.zeros(readout_size, 1)]
activ_readout = [torch.zeros(readout_size, 1) + 1e-2]
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
    cur_input = I[t] * torch.ones(lif1_size, 1)
    
    # Recurrent input
    rec_input_t = torch.matmul(conn_lif_lif, activ_filter[t-1])

    ### Input ###
    input_t = torch.mul(
        cur_input + rec_input_t,
        conn_lif_I).reshape(lif1_size, 1)

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
        activ_filter_t).reshape(readout_size, 1)
    spk_readout_t, mem_readout_t = readout(
        readout_input_t.flatten(), mem_readout[t-1].flatten())
    mem_readout_t = mem_readout_t.reshape(readout_size, 1)
    spk_readout_t = spk_readout_t.reshape(readout_size, 1)

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
        ridge_lambda = 0.1

        redout_in = torch.stack(activ_filter[-error_scope:]).clone().squeeze()
        XTX = torch.matmul(redout_in.t(), redout_in).reshape(
            lif1_size, lif1_size)
        Xy = torch.matmul(redout_in.t(), y[-error_scope:]).reshape(
            lif1_size, readout_size)
        Id = torch.eye(redout_in.shape[1]).reshape(
            lif1_size, lif1_size)
        inverse = torch.inverse(XTX + ridge_lambda * Id).reshape(
            lif1_size, lif1_size)
        conn_readout_lif = (
            inverse
            .matmul(redout_in.T)
            .matmul(y[-error_scope:])).reshape(lif1_size, readout_size)
        # print(t, ": ", conn_readout_lif)

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

added_plots = 3
tot_plots = added_plots + mem_filter.shape[1] + mem_readout.shape[1]
fig, axs = plt.subplots(tot_plots, 1, sharex=True, figsize=(7, 18))

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
    ax.set_title(f"$V_n{i}$", rotation=0)
    ax.grid()
    ax.set_xlabel(r"$t_{i}$")

for i, ax in enumerate(axs.ravel()[added_plots + mem_filter.shape[1]:]):
    if i == 0:
        ax.set_title(f"Readout")
    ax.plot(mem_readout[:, i])
    ax.set_ylabel(r"$V_{out}$", rotation=0)
    ax.grid()
    ax.set_xlabel(r"$t_{i}$")

for y_id in torch.argwhere(y):
    ax.axvline(y_id, c='r', linewidth=0.3)

fig.tight_layout()

# %% Compare spikes at the readout

fig, axs = plt.subplots(2, 1, sharex=True, figsize=(13, 4))
axs[0].plot(mem_readout[:, 0], label=r'$V_{mem}$')
axs[0].set_title(f"Readout")
axs[0].set_xlabel(r"$t_{i}$")
axs[0].grid()

axs[0].axhline(
    readout.threshold,
    linestyle='dashed',
    c='k', linewidth=1, label='threshold')
axs[0].legend()

for spk_pos in torch.argwhere(mem_readout[:, 0] >= readout.threshold):
    axs[0].axvline(spk_pos[0], linewidth=0.5, c='green')

# --
for spk_pos in torch.argwhere(mem_readout[:, 0] >= readout.threshold):
    axs[1].axvline(spk_pos[0], ymin=0.5, ymax=1, linewidth=0.3, c='green')

for y_id in torch.argwhere(y):
    axs[1].axvline(y_id, ymin=0, ymax=0.5, c='r', linewidth=0.3)

y_pred = torch.zeros_like(mem_readout).flatten()
y_pred[torch.argwhere(mem_readout[:, 0] >= readout.threshold)] = 1

y_true = torch.zeros_like(mem_readout).flatten()
y_true[torch.argwhere(y)-1] = 1

accuracy = abs(y_pred - y_true).mean()
print(f"accuracy: {accuracy:.2f}")

# %%
