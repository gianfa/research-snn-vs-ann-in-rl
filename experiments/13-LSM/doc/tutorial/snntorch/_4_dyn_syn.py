""" Single neuron dynamics with dynamic synapses

Dyn syn: time decaying synapses


"""
# %%

import snntorch as snn
import torch
import matplotlib.pyplot as plt


neuron_type  = 'Leaky'
activate_dyn_synapses = False
syn_decay = .8

# ---- Define the neuron and the input current -----

# Small step current input
# cur_in = torch.cat((torch.zeros(20), torch.ones(190) * 0.021), 0)
cur_in = torch.cat(
    [torch.cat((torch.zeros(20), torch.ones(80))) for _ in range(4)], 0)* 0.4

# Define the neuron
if neuron_type == 'Synaptic':
    # a=.35, b=.3847
    lif1 = snn.Synaptic(
    alpha=.35,  # :the decay rate of the synaptic current
    beta=.3847,  # :the decay rate of the membrane potential (as with Lapicque)
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
        alpha=.5,  # :the decay rate of the synaptic current
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
    def __init__(
        self, initial_conductance, decay_rate, learn_conductance=False):
        self.initial_conductance_ = torch.tensor(initial_conductance)
        self.conductance = torch.tensor(
            initial_conductance, requires_grad=learn_conductance)
        self.decay_rate = torch.tensor(decay_rate)

    def update(self, spk):
        # exp decay
        self.conductance = self.conductance * torch.exp(-self.decay_rate)
        # reset conductance where not spiking
        not_spiking_mask = (spk!=1).flatten()
        self.conductance[not_spiking_mask] = \
            self.initial_conductance_[not_spiking_mask].clone()
        return self.conductance
    
    def __repr__(self):
        print(f"decay_rate: {self.decay_rate}")
        return str(self.conductance)

synapse = Synapse(initial_conductance=torch.tensor([.9]), decay_rate=.0)

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
    spk_gain = 1

    if neuron_type == 'Synaptic':
        spk1, syn1, mem1 = lif1(cur_in[ti], syn1, mem1)
        act1 = spk1 * spk_gain
        syn1_rec.append(syn1)

    elif neuron_type in ['Leaky', 'Lapicque']:
        spk1, mem1 = lif1(cur_in[ti], mem1)
        if activate_dyn_synapses:
            act1 = synapse.update(spk1) * spk_gain  # more like a synapse gain
        else:
            act1 = spk1 * spk_gain

    print(f"t: {ti}; spk: {spk1}; activation: {act1}")
    if neuron_type == 'Synaptic':
        spk2, syn2, mem2 = lif2(act1, syn2, mem2)
        act2 = spk2
    elif neuron_type in ['Leaky', 'Lapicque']:
        spk2, mem2 = lif2(act1, mem2)
        act2 = spk2


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
ax.axhline(lif1.threshold, linestyle='dotted')
axs[1].set_ylabel("pre")
axs[1].grid()
for i, ax in enumerate(axs.ravel()[2:]):
    ax.plot(mem2_rec[:, 0])
    ax.set_ylabel(f"post #{i}")
    ax.axhline(lif2.threshold, linestyle='dotted')
    ax.grid()


# %%
