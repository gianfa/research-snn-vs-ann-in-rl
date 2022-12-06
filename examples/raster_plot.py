# %%
import sys
sys.path.append("..")
sys.path.append("/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/")

import torch
from stdp.funx import plot_raster

# %% Get a raster matrix

n_neurons = 8
t_steps = 30

raster = (torch.rand(n_neurons, t_steps) > 0.5).int()
raster

# %% Plot

plot_raster(raster);


# %%
