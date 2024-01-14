""" Neurons pair dynamics """
# %% Neurons pair dynamics

import snntorch as snn
import torch
import matplotlib.pyplot as plt

# ---- Define the neuron and the input current -----

# Small step current input
cur_in = torch.cat((torch.zeros(20), torch.ones(190)*0.021), 0)

# Define the neuron
lif1 = snn.Synaptic(
   alpha=0.9,  # :the decay rate of the synaptic current
   beta=0.8,  # :the decay rate of the membrane potential (as with Lapicque)
   reset_mechanism='zero'
)

# ---- Simulate ----

# History variables
mem = torch.zeros(2)  # (*) here we impose the number of neurons
syn = torch.zeros(2)
syn_rec = []  # history of synapses
mem_rec = []  # history of membrane
spk_rec = []  # history of spikes

# neuron simulation
for step in range(len(cur_in)):
  spk, syn, mem = lif1(cur_in[step], syn, mem)

  # Update the history
  mem_rec.append(mem)
  syn_rec.append(syn)
  spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
syn_rec = torch.stack(syn_rec)
spk_rec = torch.stack(spk_rec)

# ---- Visualize ----

axs = []
fig, axs = plt.subplots(mem_rec.shape[1] + 1, 1, sharex=True)
fig.suptitle("Mem Pot")
axs[0].plot(cur_in, c='orange')
axs[0].set_ylabel("I")
axs[0].grid()
for i, ax in enumerate(axs.ravel()[1:]):
    ax.plot(mem_rec[:, 0])
    ax.set_ylabel(f"#{i}")
    ax.grid()
# %%
