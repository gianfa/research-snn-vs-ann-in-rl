""" Snn.Synaptic [INCOMPLETE]

References
----------
1. https://snntorch.readthedocs.io/en/latest/snn.neurons_synaptic.html
"""
# %%
import torch
import torch.nn as nn
import snntorch as snn

alpha = 0.9
beta = 0.5

# Define Network
class Net(nn.Module):
    def __init__(
            self,
            num_inputs: int,
            num_hidden: int,
            num_outputs: int):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Synaptic(alpha=alpha, beta=beta)
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Synaptic(alpha=alpha, beta=beta)

    def forward(self, x, syn1, mem1, spk1, syn2, mem2):
        cur1 = self.fc1(x)
        spk1, syn1, mem1 = self.lif1(cur1, syn1, mem1)
        cur2 = self.fc2(spk1)
        spk2, syn2, mem2 = self.lif2(cur2, syn2, mem2)
        return syn1, mem1, spk1, syn2, mem2, spk2
    
net = Net(3, 5, 2)
net
# %%
