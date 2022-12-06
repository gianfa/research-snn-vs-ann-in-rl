# %%
""" STDP Implementation trials, on LIFs

Decription
----------
Contains tests of STDP done on LIF models.

"""
import sys
sys.path.append("..")

import snntorch as snn  # noqa
from snntorch import spikeplot as splt  # noqa
from snntorch import spikegen  # noqa

import torch  # noqa
import torch.nn as nn  # noqa
import matplotlib.pyplot as plt # noqa
from stdp.tut_utils import * # noqa
from stdp.funx import stdp_step

num_steps = 400

# layer parameters
num_inputs = 3
num_hidden = 5
num_outputs = 3
beta = 0.99

# initialize layers
fc1 = nn.Linear(num_inputs, num_hidden, bias=False)
lif1 = snn.Leaky(beta=beta)

fc2 = nn.Linear(num_hidden, num_outputs, bias=False)
lif2 = snn.Leaky(beta=beta)

# Initialize hidden states
mem1 = lif1.init_leaky()
mem2 = lif2.init_leaky()

# record outputs
cur1_rec = []
cur2_rec = []
mem1_rec = []
mem2_rec = []
spk1_rec = []
spk2_rec = []
stdp_dt_idxs = []


weights = {
    'fc2': [torch.Tensor(fc2.weight).clone()]
}

raster = spikegen.rate_conv(torch.rand((num_steps, num_inputs))).unsqueeze(1)
print(f"Dimensions of spk_in: {raster.size()}")

# %%


# network simulation
dt_stdp = 10
for step in range(num_steps):
    cur1 = fc1(raster[step])  # (1, num_inputs) x (num_inputs, num_hidden) -> 1 x num_hidden
    # (fc1) post-synaptic current <-- raster x weight
    spk1, mem1 = lif1(cur1, mem1)  # (1, num_hidden) x (num_hidden, 10) -> 1 x 10

    # (lif1) mem[t+1] <-- post-syn current + decayed membrane
    cur2 = fc2(spk1)  # (1, 10)
    spk2, mem2 = lif2(cur2, mem2)

    cur1_rec.append(cur1)
    cur2_rec.append(cur2)
    mem1_rec.append(mem1)
    mem2_rec.append(mem2)
    spk1_rec.append(spk1)
    spk2_rec.append(spk2)

    if step > 0 and step % dt_stdp == 0:
        steps = (step - dt_stdp, step)
        cur_raster = raster[steps[0]:steps[1]]
        new_weights = stdp_step(
            weights=fc2.weight,
            connections=None,
            raster=cur_raster,
            # spike_collection_rule=all_to_all,
            # dw_rule="sum",
            bidirectional=True,
            max_delta_t=20,
            inplace=False,
            v=False,
        )
        fc2.weight = nn.Parameter(new_weights.clone())

        # store the new weights
        weights['fc2'].append(new_weights.clone())
        stdp_dt_idxs.append(step)

weights = {k: torch.stack(wts) for k, wts in weights.items()}

# %%
# convert lists to tensors
cur1_rec = torch.stack(cur1_rec).squeeze().detach().numpy()
cur2_rec = torch.stack(cur2_rec).squeeze().detach().numpy()
mem1_rec = torch.stack(mem1_rec).squeeze().detach().numpy()
mem2_rec = torch.stack(mem2_rec).squeeze().detach().numpy()
spk1_rec = torch.stack(spk1_rec).squeeze()
spk2_rec = torch.stack(spk2_rec).squeeze()

# plt.plot(mem2_rec.squeeze()[:, 2].detach().numpy())

# %%

plot_cur_mem_spk(
    cur1_rec[:, 0],
    mem1_rec[:, 0],
    spk1_rec[:, 0],
    thr_line=1, ylim_max1=0.5,
    title="LIF #1")

# %%

plot_snn_spikes(
    spk_in, spk1_rec, spk2_rec, num_steps,
    "Fully Connected Spiking Neural Network")


# %%

splt.traces(mem2_rec, spk=spk2_rec)
fig = plt.gcf() 
fig.set_size_inches(8, 6)

# %%
