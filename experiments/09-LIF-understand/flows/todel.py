""""

------------



"""


import sys

import matplotlib.pyplot as plt
from pathlib import Path
import torch
import torch.nn as nn
import snntorch as snn
from sklearn.model_selection import train_test_split


sys.path += ['.', '../', '../..', '../../../']
from experimentkit_in.funx import pickle_save_dict, pickle_load
from experimentkit_in.generators.time_series import gen_lorenz
import src09





# -----

class Net(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()

        # initialize layers
        self.fc1 = nn.Linear(num_inputs, num_hidden)


        self.W = nn.Linear(num_hidden, num_hidden)
        self.lif_h = snn.Leaky(beta=beta1)

        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.inner_steps = 20

        # mem_rec = {
        #    name: [torch.zeros(1)]
        #     for name, module in net.named_modules()
        #     if isinstance(module, snn.SpikingNeuron)}
        self.mem_rec = [[], []]  # history of membrane pot
        self.spk_rec = [[], []]  # history of spikes

    def forward(self, x):                                #   t0 -> t1

        # Initialize hidden states at t=0
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        for step in range(self.inner_steps):             #   t0_0, t0_1, .., t0_10
            # Input -> hidden
            cur1 = self.fc1(x)

            # hidden -> hidden
            last_cur = None
            for step_h in range(self.inner_steps):
                cur_h = self.W(last_cur)
                spk, mem = self.lif_hidden(cur_h, mem1)

            # hidden -> output
            cur2 = self.fc2(mem1)

            # Record the layers memb and spks
            self.mem_rec[0].append(mem1)
            self.spk_rec[0].append(spk1)
            self.mem_rec[1].append(mem2)
            self.spk_rec[1].append(spk2)

        return self.mem_rec, self.spk_rec
    


