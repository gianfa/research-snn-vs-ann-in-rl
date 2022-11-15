""" STDP Implementation 2

Short description
-----------------
Implementing a more traditional STDP algorithm based on traces

Pros
----


Cons
----


"""

# %%

from typing import Iterable
import torch


def stdp_dW(
    A_plus: float, A_minus: float,
        tau_plus: float, tau_minus: float, delta_t: Iterable[int]) -> float:
    dW = torch.zeros(len(delta_t))
    dW[delta_t <= 0] = A_plus * torch.exp(delta_t[delta_t <= 0]/tau_plus)
    dW[delta_t > 0] = -A_minus * torch.exp(-delta_t[delta_t > 0]/tau_minus)
    return dW


def stdp_generate_dw_lookup(dt_max: int):
    T_lu = torch.arange(-dt_max, dt_max)
    A_plus = 0.2
    tau_plus = 5e-3
    A_minus = 0.2
    tau_minus = 4.8e-3
    dw = stdp_dW(A_plus, A_minus, tau_plus, tau_minus, T_lu*1e-5)
    T_lu = {int(dt): float(dwi) for dt, dwi in zip(T_lu.numpy().tolist(), dw)}
    T_lu[T_lu == 0] = 0
    return T_lu


n_neurons = 2
neurons_idx = torch.arange(n_neurons)

W = torch.tensor([
    [0, 0.5],
    [0.5, 0],
])
"""Weight matrix"""

A = torch.tensor([
    [0, 1],
    [-1, 0],
])
"""Adjacency matrix"""

max_delta_t = 40
T_lu = stdp_generate_dw_lookup(max_delta_t)

raster = torch.tensor([
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0]
])

expected_W = torch.tensor([
    [0, W[0, 1].item() + T_lu[-1]],
    [W[0, 1].item() + T_lu[-1], 0]
])
# %% # trial 1 - OK

v = True
i = 0
j = 1

i_spks = torch.argwhere(raster[i] == 1)
j_spks = torch.argwhere(raster[j] == 1)

for i_si in i_spks:
    diffs = (i_si - j_spks)   # all diffs between the i spk and all j spks
    diffs = diffs[diffs != 0]            # filter out zeros
    dt_ij = diffs[diffs.abs().argmin()]  # take the closer diff

    if v:
        print(f"i = {i_si.item()}")
        print(f"diffs: {(i_si - j_spks)}")
        print(f"closer dti: {dt_ij}")
        print("")
# %% # trial 2 - OK

v = True
i = 0
j = 1

spks = [torch.argwhere(raster[i] == 1) for i in neurons_idx]



# Cons: doesn't distinguish between pre and post

dts = {}
for pre in neurons_idx:
    for post in neurons_idx:
        if post == pre:
            continue
        dts[pre, post] = []
        post_spks = spks[post]
        for pre_si in spks[pre]:
            # all diffs between the i spk and all j spks
            diffs = (pre_si - post_spks)
            diffs = diffs[diffs != 0]            # filter out zeros
            dt_ij = diffs[diffs.abs().argmin()]  # take the closer diff

            dts[pre, post].append(dt_ij)
            if v:
                print(f"{pre}, {post}")
                print(f"i = {pre_si.item()}")
                print(f"diffs: {(pre_si - post_spks)}")
                print(f"closer dti: {dt_ij}")
                print("")

# %% trial 3 - OK

# for exercises purposes
# create pre_neurons and post_neurons from Adjacency matrix

A2 = torch.tensor([
    [0, 1, 1],
    [-1, 0, -1],
    [-1, 1, 0]
])



# %% ##########  IMPLEMENTATION #########
"""

"""
n_neurons = 3
neurons_idx = torch.arange(n_neurons)

W = torch.tensor([
    [0, 0.5, 0.2],
    [0.5, 0, 0.3],
    [0.15, 0, 0.7]
])
"""Weight matrix"""

raster = torch.tensor([
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0]
])

spks = [torch.argwhere(raster[i] == 1) for i in neurons_idx]
"""Spike positions for each neuron"""

connections_matrix = A2
dw_rule = "nearest_spike"  # ["sum_all", "closest_spike"]
v = True  # verbose
# %% 

# (All-to-all)
if connections_matrix is None:
    connections_matrix = torch.full_like(W, 1)
else:
    # (Nearest Neighbors)
    None
pre_post = torch.argwhere(connections_matrix > 0).numpy().tolist()
"""Pre-post pairs"""

# ## Compute all the dw ##
dts = {}
dws = {}
dts_all = {}  # stores all the `post` spikes dt
dws_all = {}  # stores all the `post` spikes dw
for pre, post in pre_post:
    if v:
        print("-->", pre, post)
    if post == pre:
        continue

    dts[pre, post] = 0
    dws[pre, post] = 0
    dts_all[pre, post] = []
    dws_all[pre, post] = []
    post_spks = spks[post]
    for pre_si in spks[pre]:
        # all diffs between the i spk and all j spks
        diffs = (pre_si - post_spks)
        diffs = diffs[diffs != 0]                   # filter out zeros
        diffs[diffs > max_delta_t] = max_delta_t    # clip - upper
        diffs[diffs < -max_delta_t] = -max_delta_t  # clip - lower

        #  Option 1: sum over all the spikes (Classical Additive)
        #    ref: eq.2 http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity#Basic_STDP_Model
        if dw_rule == "sum_all":
            dt_ij = diffs
            dw_ij = sum([T_lu[deltati.item()] for deltati in diffs])

        #  Option 2: take the closer spike (Nearest spike)
        if dw_rule == "nearest_spike":
            dt_ij = diffs[diffs.abs().argmin()].item()
            dw_ij = T_lu[dt_ij]

        # apply additive rule
        dts[pre, post] += dt_ij
        dws[pre, post] += dw_ij

        dts_all[pre, post].append(dt_ij)
        dws_all[pre, post].append(dw_ij)
        # NICETOHAVE: pre_idx, post_idx of the dt
        if v:
            print(f"{pre}, {post}")
            print(f"i = {pre_si.item()}")
            print(f"diffs: {(pre_si - post_spks)}")
            print(f"closer dti: {dt_ij}")
            print(f"dws: {dws}")
            print("")


# ## Update the weights ##
for pre_post, dw in dws.items():
    pre, post = pre_post
    W[pre, post] = W[pre, post] + dw

W


# %%
"""
Next step
--------
input to STDP flow
    - generate raster plot from a input layer taking an image as input
    - connect such a layer to STDP-layer and perform STDP
    - examine the output

test the STDP

define neurons layer
apply two stages update
    - update mem potentials
    - update weights


Description
-----------
Defines a STDP step based on the analysis of a trace of all neurons (raster)
in a given time window


Args
-----
1. Weights matrix
2. Connections matrix

3. Raster plot

"""

from typing import Iterable
import torch
from funx import *

W = torch.tensor([
    [0, 0.5, 0.2],
    [0.5, 0, 0.3],
    [0.15, 0, 0.7]
])
"""Weights matrix"""

A2 = torch.tensor([
    [0, 1, 1],
    [-1, 0, -1],
    [-1, 1, 0]
])
"""Connections matrix"""

n_neurons = W.shape[0]
neurons_idx = torch.arange(n_neurons)


raster = torch.tensor([
    [0, 1, 0, 1, 0, 0],
    [0, 0, 1, 1, 0, 0],
    [0, 0, 1, 1, 0, 0]
])

spks = [torch.argwhere(raster[i] == 1) for i in neurons_idx]
"""Spike positions for each neuron"""

connections_matrix = A2
dw_rule = "nearest_spike"  # ["sum_all", "closest_spike"]
v = False  # verbose

# --
# (Nearest Neighbors)
connections_matrix = A2
# (All-to-all)
if connections_matrix is None:
    connections_matrix = torch.full_like(W, 1)
pre_post = torch.argwhere(connections_matrix > 0).numpy().tolist()
"""Pre-post pairs"""

# ## Compute all the dw ##
dts = {}
dws = {}
dts_all = {}  # stores all the `post` spikes dt
dws_all = {}  # stores all the `post` spikes dw
for pre, post in pre_post:
    if v:
        print("-->", pre, post)
    if post == pre:
        continue

    dts[pre, post] = 0
    dws[pre, post] = 0
    dts_all[pre, post] = []
    dws_all[pre, post] = []
    post_spks = spks[post]
    for pre_si in spks[pre]:
        # all diffs between the i spk and all j spks
        diffs = (pre_si - post_spks)
        diffs = diffs[diffs != 0]                   # filter out zeros
        diffs[diffs > max_delta_t] = max_delta_t    # clip - upper
        diffs[diffs < -max_delta_t] = -max_delta_t  # clip - lower

        #  Option 1: sum over all the spikes (Classical Additive)
        #    ref: eq.2 http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity#Basic_STDP_Model
        if dw_rule == "sum_all":
            dt_ij = diffs
            dw_ij = sum([T_lu[deltati.item()] for deltati in diffs])

        #  Option 2: take the closer spike (Nearest spike)
        if dw_rule == "nearest_spike":
            dt_ij = diffs[diffs.abs().argmin()].item()
            dw_ij = T_lu[dt_ij]

        # apply additive rule
        dts[pre, post] += dt_ij
        dws[pre, post] += dw_ij

        dts_all[pre, post].append(dt_ij)
        dws_all[pre, post].append(dw_ij)
        # NICETOHAVE: pre_idx, post_idx of the dt
        if v:
            print(f"{pre}, {post}")
            print(f"i = {pre_si.item()}")
            print(f"diffs: {(pre_si - post_spks)}")
            print(f"closer dti: {dt_ij}")
            print(f"dws: {dws}")
            print("")


# ## Update the weights ##
for pre_post, dw in dws.items():
    pre, post = pre_post
    W[pre, post] = W[pre, post] + dw

W

# %%


# %%
import snntorch as snn  # noqa
import torch  # noqa
from snntorch import spikegen  # noqa
from funx import *


raster = torch.tensor([
        [0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 0],
        [0, 0, 1, 1, 1, 0],
        [1,1,1,1,1,1]
    ])

plot_raster(
    raster=raster,
    dt = 1e3, # [s]
    show_horizontal_lines = False
)
# %%
