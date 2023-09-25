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

"""
# %%

import snntorch as snn
import torch
import matplotlib.pyplot as plt

# ---- Define the neuron and the input current -----

# Small step current input
cur_in = torch.cat((torch.zeros(20), torch.ones(190)*0.2), 0)

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
    state_quant=False,  # If `True` as well as `init_hidden=True`, states are returned
    output=False
   )

# ---- Simulate ----

# History variables
mem = torch.zeros(1)
mem_rec = []  # history of membrane
spk_rec = []  # history of spikes

# neuron simulation
for step in range(len(cur_in)):
  spk, mem = lif1(cur_in[step], mem)

  # Update the history
  mem_rec.append(mem)
  spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
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
