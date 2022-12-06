# %%
import sys
sys.path.append("..")
sys.path.append("/Users/giana/Desktop/Github/Projects/research-snn-vs-ann-in-rl/")
from typing import Callable, Dict, Iterable, List, Tuple
import numpy as np
import string
from typing import Iterable, List, Tuple
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import snntorch as snn
import torch

from stdp.spike_collectors import all_to_all, nearest_pre_post_pair
from stdp.funx import (plot_raster, get_spike_positions_from_tpre_tpost,
    get_raster_from_spike_positions, get_equispaced_rgbas_from_cmap,
    get_raster_from_tpre_tpost, connections_to_digraph, 
    plot_composed_rasters_from_tpre_tpost_groups, raster_collect_spikes)



# %%

raster = (torch.rand(5, 10) > 0.35).int()
# raster = torch.Tensor([[ True, False,  True, False,  True, False,  True, False,  True, False],
#         [ True,  True, False,  True,  True, False,  True,  True, False,  True],
#         [ True, False,  True,  True, False,  True, False, False,  True,  True],
#         [False, False,  True, False, False,  True,  True, False, False, False],
#         [ True, False,  True,  True, False, False, False, False, False,  True]])

raster
# %%

plot_raster(raster)

# %%

connections = torch.Tensor([
    [0, 1, 0, 0, 0],
    [0, 0, 1, 1, 0],
    [1, 0, 0, 0, 0],
    [0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0]
])

fig, ax = plt.subplots(1, 1)
ax.set_title("Connections")
G, ax = connections_to_digraph(connections, ax=ax)

# %%

tpre_tpost_groups = raster_collect_spikes(
    raster, nearest_pre_post_pair, connections)

tpre_tpost_groups


# %%
raster_rows, ax = plot_raster(raster)
ax.figure

# %%
ax = plot_composed_rasters_from_tpre_tpost_groups(
    tpre_tpost_groups, size=raster.shape, ax=ax)
ax.figure

# %%
