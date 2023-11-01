""" Single neuron dynamics

3 main parts:
1. Neuron and input definition
2. Simulation
3. Visualization


It's all about two main parts: the neuron loop and how to feed it.
Since the neuron has a temporal memory, one should pass the last time step
state to the current time step. This happens during the loop.
```python
for step in range(t_steps):
  spk, mem = neuron(prev_activation, mem)
  prev_activation = f(mem)
```

Inside the loop, the neuronal feeding happens through passing the previous 
activation as the current input.


The input current is weighted by (1 - beta).
(1-beta) I_in = I_in/tau .[Eshragian](https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_3.html)

"""
# %%

import snntorch as snn
import torch
import matplotlib.pyplot as plt


neuron_type  = 'Lapicque'

# ---- Define the neuron and the input current -----

# Small step current input
# cur_in = torch.cat((torch.zeros(20), torch.ones(190) * 0.021), 0)
cur_in = torch.cat(
    [torch.cat((torch.zeros(20), torch.ones(50))) for _ in range(6)], 0)* 0.4

# Define the neuron
if neuron_type == 'Synaptic':
    lif1 = snn.Synaptic(
    alpha=.3,  # :the decay rate of the synaptic current
    beta=0.4,  # :the decay rate of the membrane potential (as with Lapicque)
    reset_mechanism='zero'
    )
elif neuron_type == 'Leaky':
    lif1 = snn.Leaky(
    beta=0.8,  # :the decay rate of the membrane potential (as with Lapicque)
    reset_mechanism='zero'
    )
elif neuron_type == 'Lapicque':
    lif1 = snn.Lapicque(
    R=.3e1,
    C=6e-1,
    time_step=1e-1,
    # beta=0.8,
    threshold=1,
    reset_mechanism='zero'
    )

# lif1 = snn.Leaky(beta=0.85)


# ---- Simulate ----

# History variables
mem = torch.zeros(1)
syn = torch.zeros(1)
syn_rec = []  # history of synapses
mem_rec = []  # history of membrane
spk_rec = []  # history of spikes

# neuron simulation
for step in range(len(cur_in)):
    if neuron_type == 'Synaptic':
        spk, syn, mem = lif1(cur_in[step], syn, mem)
        syn_rec.append(syn)

    elif neuron_type in ['Leaky', 'Lapicque']:
        spk, mem = lif1(cur_in[step], mem)
    
    # Update the history
    mem_rec.append(mem)
    spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)
if neuron_type == 'Synaptic':
    syn_rec = torch.stack(syn_rec)

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
ax.axhline(lif1.threshold, linestyle='dotted')


# %%
