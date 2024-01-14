""" Single neuron dynamics with dynamic synapses

Dyn syn: time decaying synapses


"""
# %%
import sys

import snntorch as snn
import torch
import matplotlib.pyplot as plt

sys.path += ['../../../../../']
from experimentkit_in.collections import FIFO_buffer

neuron_type  = 'Leaky'
activate_dyn_synapses = True
syn_initial_conductance = .9
syn_decay = .5
# syn_recovery = 0

# ---- Define the neuron and the input current -----

# Small step current input
# cur_in = torch.cat((torch.zeros(20), torch.ones(190) * 0.021), 0)
cur_in = torch.cat(
    [torch.cat((torch.zeros(20), torch.ones(80))) for _ in range(4)], 0)* 0.4

# Define the neuron
if neuron_type == 'Synaptic':
    lif1 = snn.Synaptic(
    alpha=.3,  # :the decay rate of the synaptic current
    beta=0.4,  # :the decay rate of the membrane potential (as with Lapicque)
    reset_mechanism='zero'
    )
elif neuron_type == 'Leaky':
    lif1 = snn.Leaky(
    beta=0.6008,  # :the decay rate of the membrane potential (as with Lapicque)
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


if neuron_type == 'Synaptic':
    lif2 = snn.Synaptic(
        alpha=.3,  # :the decay rate of the synaptic current
        beta=0.4,  # :the decay rate of the membrane potential (as with Lapicque)
        reset_mechanism='zero'
    )
elif neuron_type == 'Leaky':
    lif2 = snn.Leaky(
        beta=0.95,  # :the decay rate of the membrane potential (as with Lapicque)
        reset_mechanism='zero'
    )
elif neuron_type == 'Lapicque':
    lif2 = snn.Lapicque(
        R=.3e1,
        C=6e-1,
        time_step=1e-1,
        # beta=0.8,
        threshold=1,
        reset_mechanism='zero'
    )


class Synapse:
    """Synapse class

    Parameters
    ----------
    initial_conductance : _type_
        _description_
    decay_rate : _type_
        _description_
    learn_conductance : bool, optional
        _description_, by default False

    Example
    -------
    >>> size = 5
    >>> initial_conductance = 0.8
    >>> decay_rate = 0.9
    >>> synapse = Synapse((5, 5), initial_conductance, decay_rate)
    >>> spike_history1 = torch.tensor([[1], [0], [1], [1], [0]], dtype=torch.bool)
    >>> spike_history2 = torch.tensor([[0], [1], [1], [1], [1]], dtype=torch.bool)
    >>> print(spike_history)
    >>> synapse.update(spike_history1)
    >>> print(synapse.synaptic_matrix)
    >>> synapse.update(spike_history2)
    >>> print(synapse.synaptic_matrix)
    """
    def __init__(
        self,
        size: tuple,
        initial_conductance: torch.Tensor,
        decay_rate: torch.Tensor,
        learn_conductance: bool = False):
        self.initial_conductance_ = initial_conductance
        self.synaptic_matrix = torch.ones(size) * initial_conductance
        self.decay_rate = decay_rate

    def update(self, spk: torch.Tensor):
        
        # mask of the neurons who had a spike
        spiked_neurons = spk.view(-1).nonzero().squeeze()
        # synaptic decay for firing neurons
        self.synaptic_matrix[spiked_neurons] *= self.decay_rate

        # reset conductance where not spiking
        size = self.synaptic_matrix.size(0)
        non_spiked_neurons = torch.ones(size, dtype=torch.bool)
        non_spiked_neurons[spiked_neurons] = False
        # recovery
        self.synaptic_matrix[non_spiked_neurons] = 1.0
        return self.synaptic_matrix
    
    def __repr__(self):
        print(f"decay_rate: {self.decay_rate}")
        return str(self.synaptic_matrix)


synapse = Synapse(
    size=(1),
    initial_conductance=torch.tensor([syn_initial_conductance]),
    decay_rate=torch.tensor(syn_decay))


# ---- Simulate ----



# History variables
mem1 = torch.zeros(1)
spk1 = torch.zeros(1)
mem2 = torch.zeros(1)
syn1 = torch.zeros(1)
syn2 = torch.zeros(1)
prev_act = torch.ones(1)

syn1_rec = []  # history of synapses
mem1_rec = []  # history of membrane
spk1_rec = []  # history of spikes
act1_rec = []  # history of activations
syn2_rec = []
mem2_rec = []
spk2_rec = []
act2_rec = []
prev_act_rec = []



# neuron simulation
for ti in range(len(cur_in)):
    if neuron_type == 'Synaptic':
        spk, syn, mem1 = lif1(cur_in[ti], syn, mem1)
        syn1_rec.append(syn)

    elif neuron_type in ['Leaky', 'Lapicque']:
        spk_gain = 1.1
        
        spk1, mem1 = lif1(cur_in[ti], mem1)
        if activate_dyn_synapses:
            synapse.update(spk1)
            act1 = spk1 * synapse.synaptic_matrix * spk_gain  # more like a synapse gain
        else:
            act1 = spk1 * spk_gain

        print(f"t: {ti}; spk: {spk1}; activation: {act1}")
        spk2, mem2 = lif2(act1, mem2)
        act2 = mem2


    # Update the history
    mem1_rec.append(mem1)
    spk1_rec.append(spk1)
    act1_rec.append(act1)
    mem2_rec.append(mem2)
    spk2_rec.append(spk2)
    prev_act_rec.append(prev_act)

# convert lists to tensors
mem1_rec = torch.stack(mem1_rec)
spk1_rec = torch.stack(spk1_rec)
mem2_rec = torch.stack(mem2_rec)
spk2_rec = torch.stack(spk2_rec)
if neuron_type == 'Synaptic':
    syn1_rec = torch.stack(syn1_rec)

# ---- Visualize ----

axs = []
fig, axs = plt.subplots(mem1_rec.shape[1] + 2, 1, sharex=True)
fig.suptitle("Mem Pot")
axs[0].plot(cur_in, c='orange')
axs[0].set_ylabel("I")
axs[0].grid()
axs[1].plot(mem1_rec[:, 0])
axs[1].set_ylabel("pre")
axs[1].grid()
for i, ax in enumerate(axs.ravel()[2:]):
    ax.plot(mem2_rec[:, 0])
    ax.set_ylabel(f"post #{i}")
    ax.grid()
ax.axhline(lif1.threshold, linestyle='dotted')


# %%
