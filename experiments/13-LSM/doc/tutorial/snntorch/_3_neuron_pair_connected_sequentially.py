""" Connected Neurons pair dynamics - Synaptic

I -> LIF_1 -> LIF_2 ->


"""
# %% Neurons pair dynamics

import snntorch as snn
import torch
import matplotlib.pyplot as plt

neuron_type = 'Leaky'

# ---- Define the neuron and the input current -----

# def test_1():
# Small step current input
I = torch.cat(
    (torch.zeros(20),
    torch.ones(110) * 0.2,
    torch.zeros(80)), dim=0)

if neuron_type == 'Synaptic':
    # Define the neuron
    lif1 = snn.Synaptic(
    alpha=0.9,  # :the decay rate of the synaptic current
    beta=0.8,  # :the decay rate of the membrane potential (as with Lapicque)
    reset_mechanism='zero'
    )
    lif2 = snn.Synaptic(
    alpha=0.9,  # :the decay rate of the synaptic current
    beta=0.8,  # :the decay rate of the membrane potential (as with Lapicque)
    reset_mechanism='zero'
    )

elif neuron_type == 'Leaky':
    # Define the neuron
    lif1 = snn.Leaky(
    beta=0.8,  # :the decay rate of the membrane potential (as with Lapicque)
    reset_mechanism='zero'
    )
    lif2 = snn.Leaky(
    beta=0.8,  # :the decay rate of the membrane potential (as with Lapicque)
    reset_mechanism='zero'
    )

def thresholded_relu(x, threshold):
    return torch.where(x > threshold, x, 0)

# ---- Simulate ----

# History variables
mem_1 = [torch.zeros(1, 1)]
spk_1 = [torch.zeros(1, 1)]
syn_1 = [torch.zeros(1, 1)]
activ_1 = [torch.zeros(1, 1)]
mem_2 = [torch.zeros(1, 1)]
spk_2 = [torch.zeros(1, 1)]
syn_2 = [torch.zeros(1, 1)]
activ_2 = [torch.zeros(1, 1)]

input = []

# %%
# neuron simulation
for t in range(1, len(I) -1):
    # Current input: Hadamard I . conn_lif_I
    cur_input = I[t] * torch.ones(1, 1)

    input_t = cur_input

    if neuron_type == 'Synaptic':
        spk_1_t, syn_1_t, mem_1_t = lif1(
            input_t.squeeze(), syn_1[t-1], mem_1[t-1].squeeze())
    elif neuron_type == 'Leaky':        
        spk_1_t, mem_1_t = lif1(
            input_t.squeeze(), mem_1[t-1].squeeze())
    mem_1_t = mem_1_t.reshape(1, 1)  # make it columnar
    spk_1_t = spk_1_t.reshape(1, 1)  # make it columnar

    activ_1_t = spk_1_t

    if neuron_type == 'Synaptic':
        spk_2_t, syn_2_t, mem_2_t = lif1(
            activ_1_t, syn_2[t-1], mem_2[t-1].squeeze())
    elif neuron_type == 'Leaky':
        spk_2_t, mem_2_t = lif1(
            activ_1_t, mem_2[t-1].squeeze())

    mem_2_t = mem_2_t.reshape(1, 1)  # make it columnar
    spk_2_t = spk_2_t.reshape(1, 1)  # make it columnar

    activ_2_t = (1 * mem_2_t)

    # store current state
    input.append(input_t)

    mem_1.append(mem_1_t)
    spk_1.append(spk_1_t)
    syn_1.append(syn_1_t)
    activ_1.append(activ_1_t)

    mem_2.append(mem_2_t)
    spk_2.append(spk_2_t)
    syn_2.append(syn_2_t)
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
