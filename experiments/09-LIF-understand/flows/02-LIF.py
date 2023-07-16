""" LIF & LIF Networks


It is assumed that:
- the behaviour of LIF neurons is approximated by activation functions, which
    specifically implement the typical integration of LIFs.
- each layer of weights (Linear, Convolutional, ..) is considered a matrix of
    synaptic connections
- The product operation between the input travelling and the matrix of
    weights returns the input current to the neurons.

    
During Training
- At each step the x_i example is shown to the network.
    For all the internal dynamics execution, namely during all the 
    internal loop, the same x_i is shown to the network.

References
----------
1. https://snntorch.readthedocs.io/en/latest/snn.neurons_leaky.html
2. https://colab.research.google.com/github/jeshraghian/snntorch/blob/master/examples/tutorial_2_lif_neuron.ipynb
"""
# %%
import sys

import matplotlib.pyplot as plt
import snntorch as snn
import torch
import torch.nn as nn

sys.path += ['.', '../']
import src09

# %%  --------------- Test a Leaky neuron model ---------------

num_steps = 100  # simulation t steps
beta = 0.8

# Small step current input
cur_in = torch.cat((torch.zeros(10), torch.ones(90)*0.8), 0)


mem = torch.zeros(1)
mem_rec = []  # history of membrane
spk_rec = []  # history of spikes

lif1 = snn.Leaky(
    beta=beta,  # membrane potential decay rate, clipped. May be a tensor.
    threshold=1,
    reset_mechanism="subtract",  # {'subtract', 'zero', 'none'}
    spike_grad=None,  # surrogate gradient function
    init_hidden=False,
    inhibition=False,  # If `True`, suppresses all spiking other than the neuron with the highest state
    learn_beta=False,  # enable learnable beta
    learn_threshold=False,  # enable learnable threshold
    state_quant=False,  # If `True` as well as `init_hidden=True`, states are returned
    output=False
)

# neuron simulation
for step in range(num_steps):
  mem, spk = lif1(cur_in[step], mem)
  mem_rec.append(mem)
  spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)

# Plot
# src09.utils.plot_cur_mem_spk(
#     cur_in, mem_rec, spk_rec,
#     thr_line=1, vline=109, ylim_mempot=(0, 1.3), 
#     title="LIF Neuron Model")

fig, axs = plt.subplots(2)
axs[0].plot(cur_in, color='orange')
axs[1].plot(mem_rec)

# %% --------------- Test a Lapique neuron model ---------------

num_steps = 20  # simulation t steps

# Small step current input
cur_in = torch.cat((torch.zeros(10), torch.ones(190)*0.2), 0)


mem = torch.zeros(1)
mem_rec = []  # history of membrane
spk_rec = []  # history of spikes

lif1 = snn.Lapicque(
    beta=False,
    R=5.1,
    C=0.01,
    time_step=0.5,
    threshold=0.5,
    spike_grad=None,  # surrogate gradient function
    init_hidden=False,
    inhibition=False,  # If `True`, suppresses all spiking other than the neuron with the highest state
    learn_beta=False,  # enable learnable beta
    learn_threshold=False,  # enable learnable threshold
    reset_mechanism="subtract",  # {'subtract', zero', 'none'}
    state_quant=False,  # If `True` as well as `init_hidden=True`, states are returned
    output=False
   )
lif1 =snn.Lapicque(
    R=5.1, C=0.005, time_step=0.001, threshold=0.5,
    reset_mechanism='subtract')
# neuron simulation
for step in range(len(cur_in)):
  spk, mem = lif1(cur_in[step], mem)
  mem_rec.append(mem)
  spk_rec.append(spk)

# convert lists to tensors
mem_rec = torch.stack(mem_rec)
spk_rec = torch.stack(spk_rec)


# Visualize
range_to_plot = slice(0, 150) # len(net.mem_rec[0])

fig, axs = plt.subplots(3)
axs[0].plot(cur_in[range_to_plot], label=f"I_in", color='orange')
axs[0].set_title('$I_{in}$')
axs[0].set_xlim(range_to_plot.start, range_to_plot.stop)
axs[0].grid()

axs[1].plot(mem_rec[range_to_plot, 0], label=f"n 0")
axs[1].axhline(lif1.threshold, linestyle='dotted', color='grey', label='lif1')
axs[1].set_ylim(-0.5, 1.2)
axs[1].set_title('$V_{mem}$')
axs[1].set_xlim(range_to_plot.start, range_to_plot.stop)
axs[1].grid()
axs[1].legend()

spk_0_i = torch.argwhere(spk_rec[range_to_plot, 0].flatten())
axs[2].scatter(spk_0_i, torch.ones_like(spk_0_i), label=f"n 0", marker='|')
axs[2].set_xlim(range_to_plot.start, range_to_plot.stop)
fig.tight_layout()
# Plot
# src09.utils.plot_cur_mem_spk(
#     cur_in, mem_rec, spk_rec,
#     thr_line=0.5, ylim_mempot=(0, 1.25), 
#     title="Lapique Neuron Model")

# %% ---------------  Two neurons model ---------------  

num_steps = 100  # simulation t steps
beta = 0.05

# Small step current input
cur_in = torch.cat(
   (
      torch.zeros(10),
      torch.ones(120)*0.7,
      torch.ones(70)*0.2
    ), 0)


mem1 = torch.zeros(1)
mem2 = torch.zeros(1)
mem_rec = [[], []]  # history of membrane
spk_rec = [[], []]  # history of spikes

lif1 = snn.Leaky(
    beta=beta,  # membrane potential decay rate, clipped. May be a tensor.
    threshold=1,
    spike_grad=None,  # surrogate gradient function
    init_hidden=False,
    inhibition=False,  # If `True`, suppresses all spiking other than the neuron with the highest state
    learn_beta=False,  # enable learnable beta
    learn_threshold=False,  # enable learnable threshold
    reset_mechanism="zero",  # {'subtract', zero', 'none'}
    state_quant=False,  # If `True` as well as `init_hidden=True`, states are returned
    output=False
)

lif2 = snn.Leaky(
    beta=beta,  # membrane potential decay rate, clipped. May be a tensor.
    threshold=1,
    spike_grad=None,  # surrogate gradient function
    init_hidden=False,
    inhibition=False,  # If `True`, suppresses all spiking other than the neuron with the highest state
    learn_beta=False,  # enable learnable beta
    learn_threshold=False,  # enable learnable threshold
    reset_mechanism="zero",  # {'subtract', zero', 'none'}
    state_quant=False,  # If `True` as well as `init_hidden=True`, states are returned
    output=False
)

# neuron simulation
for step in range(len(cur_in)):
  cur_in1 = cur_in[step]
  spk1, mem1 = lif1(cur_in1, mem1)

  cur_in2_1 = torch.tensor([0.1]) * mem1
  spk2, mem2 = lif2(cur_in2_1, mem2)

  mem_rec[0].append(mem1)
  mem_rec[1].append(mem2)
  spk_rec[0].append(spk1)
  spk_rec[1].append(spk2)

# convert lists to tensors
mem_rec = [torch.stack(mi) for mi in mem_rec]
spk_rec = [torch.stack(sri) for sri in spk_rec]


# %% Visualize
range_to_plot = slice(0, 150) # len(net.mem_rec[0])

fig, axs = plt.subplots(4, figsize=(10,10))
axs[0].plot(cur_in[range_to_plot], label=f"I_in", color='orange')
axs[0].set_title('$I_{in}$')
axs[0].set_xlim(range_to_plot.start, range_to_plot.stop)
axs[0].grid()

axs[1].plot(mem_rec[0][range_to_plot, 0])
axs[1].axhline(
  lif1.threshold, linestyle='dotted', color='grey', label="threshold")
axs[1].set_ylim(-0.5, 1.2)
axs[1].set_title('LIF1 $V_{mem}$')
axs[1].set_xlim(range_to_plot.start, range_to_plot.stop)
axs[1].grid()
axs[1].legend()

axs[2].plot(mem_rec[1][range_to_plot, 0])
axs[2].axhline(
   lif1.threshold, linestyle='dotted', color='grey', label="threshold")
axs[2].set_ylim(-0.5, 1.8)
axs[2].set_title('LIF2 $V_{mem}$')
axs[2].set_xlim(range_to_plot.start, range_to_plot.stop)
axs[2].grid()
axs[2].legend()

# spikes. TODO: a plot_spikes function
for i in range(len(spk_rec)):
    spk_0_i = torch.argwhere(spk_rec[i][range_to_plot, 0].flatten())
    axs[3].scatter(
       spk_0_i,
       torch.ones_like(spk_0_i) * (len(spk_rec)-i),
       label=f"n 0", marker='|', s=int(6e2))
axs[3].set_xlim(range_to_plot.start, range_to_plot.stop)
axs[3].set_ylim(0.5, len(spk_rec)+0.5)
axs[3].set_title("Spike events")
# TODO: add var y_labels: (LIF1, LIF2,...)
fig.tight_layout()

# Plot
# n_i = 0
# src09.utils.plot_cur_mem_spk(
#     cur_in, mem_rec[n_i], spk_rec[n_i],
#     thr_line=1, vline=109, ylim_mempot=(0, 1.3), 
#     title="LIF Neuron Model")


# %%

# Based on
# https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html

beta1 = 9
beta2 = 0.5


# Define Network
class Net(nn.Module):
    def __init__(
          self, num_inputs, num_hidden, num_outputs,
            inner_steps=1,  
        ):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta1)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2)
        self.inner_steps = inner_steps

        # mem_rec = {
        #    name: [torch.zeros(1)]
        #     for name, module in net.named_modules()
        #     if isinstance(module, snn.SpikingNeuron)}
        self.mem_rec = [[], []]  # history of membrane pot
        self.spk_rec = [[], []]  # history of spikes

    def forward(self, x):
        """
        At each step the x_i example is shown to the network.
        For all the internal dynamics execution, namely during all the 
        internal loop, the same x_i is shown to the network.
        """

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(self.inner_steps):
            cur1 = self.fc1(x)
            spk1, mem1 = self.lif1(cur1, mem1)

            cur2 = self.fc2(mem1)
            spk2, mem2 = self.lif2(cur2, mem2)

            # Record the layers memb and spks
            self.mem_rec[0].append(mem1.detach())
            self.spk_rec[0].append(spk1.detach())
            self.mem_rec[1].append(mem2.detach())
            self.spk_rec[1].append(spk2.detach())

        return mem1, mem2

# %%
# Small step current input
cur_in = torch.cat((torch.zeros(10), torch.ones(190)*8.2), 0)[:, None]


net = Net(1, 2, 1, inner_steps=1)
for cur_step in cur_in:
   net(cur_step.view(-1))


net.mem_rec[0] = torch.stack(net.mem_rec[0])
net.mem_rec[1] = torch.stack(net.mem_rec[1])
net.spk_rec[0] = torch.stack(net.spk_rec[0])
net.spk_rec[1] = torch.stack(net.spk_rec[1])

print(f"cur_in.shape: {cur_in.shape}")
print(f"num_steps: {num_steps}")
print(f"net.inner_steps: {net.inner_steps}")
print(f"net.mem_rec[0].shape: {net.mem_rec[0].shape}")
print(f"net.mem_rec[1].shape: {net.mem_rec[1].shape}")

print(f"net.lif1.threshold: {net.lif1.threshold}")
print(f"net.lif2.threshold: {net.lif2.threshold}")




import matplotlib.pyplot as plt

range_to_plot = slice(0, 50) # len(net.mem_rec[0])

fig, axs = plt.subplots(4)
axs[0].plot(cur_in[range_to_plot], label=f"I_in", color='orange')
axs[0].grid()

axs[1].plot(net.mem_rec[0][range_to_plot, 0], label=f"n 0")
axs[1].plot(net.mem_rec[0][range_to_plot, 1], label=f"n 1", c='green')
axs[1].axhline(
   net.lif1.threshold, linestyle='dotted', color='grey', label='0-th')
axs[1].grid()
axs[1].legend()

spk_0_i = torch.argwhere(net.spk_rec[0][range_to_plot, 0].flatten())
axs[2].scatter(spk_0_i, torch.ones_like(spk_0_i), label=f"n 0", marker='|')
axs[2].grid()
axs[2].legend()

axs[3].plot(net.mem_rec[1][range_to_plot, 0])
axs[3].grid()
axs[3].legend()


# %% Plot

n_i = 0
src09.utils.plot_cur_mem_spk(
    cur_in,
    torch.stack(net.mem_rec[n_i]).detach(),
    torch.stack(net.spk_rec[n_i]).detach(),
    thr_line=1, ylim_mempot=(0, 1.25), 
    title=f"LIF Layer #{n_i}")
