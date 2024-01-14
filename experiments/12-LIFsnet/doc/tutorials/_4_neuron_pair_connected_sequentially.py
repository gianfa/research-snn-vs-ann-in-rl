""" Connected Neurons pair dynamics

I -> LIF_1 -> LIF_2 ->


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
    torch.ones(110) * 0.4,
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

def thresholded_relu(x, threshold):
    return torch.where(x > threshold, x, 0)

# ---- Simulate ----

# History variables
mem_1 = [torch.zeros(1, 1)]
spk_1 = [torch.zeros(1, 1)]
activ_1 = [torch.zeros(1, 1)]
mem_2 = [torch.zeros(1, 1)]
spk_2 = [torch.zeros(1, 1)]
activ_2 = [torch.zeros(1, 1)]

input = []

# %%
# neuron simulation
for t in range(1, len(I) -1):
    # Current input: Hadamard I . conn_lif_I
    cur_input = I[t] * torch.ones(1, 1)

    input_t = cur_input

    spk_1_t, mem_1_t = lif1( input_t.squeeze(), mem_1[t-1].squeeze())
    mem_1_t = mem_1_t.reshape(1, 1)  # make it columnar
    spk_1_t = spk_1_t.reshape(1, 1)  # make it columnar

    activ_1_t = thresholded_relu(1 * mem_1_t, lif1.threshold)

    spk_2_t, mem_2_t = lif1( activ_1_t, mem_2[t-1].squeeze())
    mem_2_t = mem_2_t.reshape(1, 1)  # make it columnar
    spk_2_t = spk_2_t.reshape(1, 1)  # make it columnar

    activ_2_t = (1 * mem_2_t)

    # store current state
    input.append(input_t)

    mem_1.append(mem_1_t)
    spk_1.append(spk_1_t)
    activ_1.append(activ_1_t)

    mem_2.append(mem_2_t)
    spk_2.append(spk_2_t)
    activ_2.append(activ_2_t)


# convert lists to tensors
input = torch.stack(input)
mem_1 = torch.stack(mem_1)
spk_1 = torch.stack(spk_1)
activ_1 = torch.stack(activ_1)
mem_2 = torch.stack(mem_2)
spk_2 = torch.stack(spk_2)
activ_2 = torch.stack(activ_2)

# ---- Visualize ----
axs = []
fig, axs = plt.subplots(3, 1, sharex=True)

axs[0].plot(I, c="red")
axs[0].set_ylabel(f"$I$", rotation=0)
axs[0].grid()

axs[1].plot(mem_1.squeeze())
axs[1].set_ylabel(r"$V_{n1}$")
axs[1].grid()

axs[2].plot(mem_2.squeeze())
axs[2].set_ylabel(r"$V_{n2}$")
axs[2].grid()


# %%
