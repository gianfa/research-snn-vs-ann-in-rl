""" Training a Network



References
----------
1. https://snntorch.readthedocs.io/en/latest/tutorials/tutorial_5.html
"""
# %% Define a network
from functools import partial
import sys

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import snntorch as snn

sys.path += ['.', '../']
import src09

beta1 = 0.5
beta2 = 0.5

# %% Prepare the containers for recording mem pot and spike hisotries

mem_rec = {  # will collect membrane potentials
    'lif1': [],
    'lif2': [],
}
spk_rec = {  # will collect spikes
    'lif1': [],
    'lif2': [],
}

def record_outputs(layer, input, output, label: str):
    spk, mem = output
    mem_rec[label].append(mem)
    spk_rec[label].append(spk)

# %%

# Define Network
class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta1)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta2)
        self.inner_steps = 10

        # mem_rec = {
        #    name: [torch.zeros(1)]
        #     for name, module in net.named_modules()
        #     if isinstance(module, snn.SpikingNeuron)}

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

        return spk2, mem2

# %% Training

# exp params
dtype = torch.float
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Small step current input
cur_in = torch.cat(
   (torch.zeros(10),
    torch.ones(20) * 0.2,
    torch.ones(40) * 0.8,
    torch.ones(30) * 0.2,
    torch.ones(30) * 0.4,
    torch.ones(70) * 0.2),
   0)[:, None]
targets = ((cur_in > 0.2) * (cur_in < 0.8)).float()

bs = 50
X = cur_in.reshape(cur_in.shape[0]//bs, bs)
Y = targets.reshape(targets.shape[0]//bs, bs)

net = Net(bs, 2, bs)

# %% Register hooks to the net

if 'hooks' in locals():
    for hook in hooks:
        hook.remove()

hooks=[]
hooks.append(
    net.lif1.register_forward_hook(partial(record_outputs, label='lif1')))
hooks.append(
    net.lif2.register_forward_hook(partial(record_outputs, label='lif2')))

# %%


criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=5e-3, betas=(0.9, 0.999))

num_epochs = 100
loss_hist = []
for epoch in range(num_epochs):
    iter_counter = 0

    # batch-wise training
    for xi, yi in zip(X, Y):
        optimizer.zero_grad()
        net.train()
        spk, mem = net(xi)

        # initialize the loss & sum over time
        # loss_val = torch.zeros((1), dtype=dtype, device=device)
        # for step in range(net.inner_steps):
        #     loss_val += criterion(mem_rec[1][step], yi)
        loss_val = sum([
           criterion(mem.squeeze(), yi)
           for step in range(net.inner_steps)])
        # loss_val = criterion(mem[1][0], yi)
        loss_hist.append(loss_val.item())

        # Gradient calculation + weight update
        loss_val.backward()
        optimizer.step()

        iter_counter +=1
        print(f"Training loss: {loss_val}")
        print(f"fc2 weights: {net.fc2.weight.data.mean()}")

fig, ax = plt.subplots()
ax.plot(range(len(loss_hist)), loss_hist, label='Training Loss')
ax.legend()

# %%


print(f"cur_in.shape: {cur_in.shape}")
print(f"num_steps: {net.inner_steps}")
print(f"net.inner_steps: {net.inner_steps}")

# Hooks remove
if 'hooks' in locals():
    for hook in hooks:
        hook.remove()
    del hooks

# stack the recs
mem_rec['lif1'] = torch.stack(mem_rec['lif1'])
mem_rec['lif2'] = torch.stack(mem_rec['lif2'])
spk_rec['lif1'] = torch.stack(spk_rec['lif1'])
spk_rec['lif2'] = torch.stack(spk_rec['lif2'])

# See the mem and spk histories
for layer_name, mem in mem_rec.items():
    print(f"{layer_name} mem_rec: {mem_rec[layer_name].shape}")
    print(f"{layer_name} spk_rec: {spk_rec[layer_name].shape}")


# %%