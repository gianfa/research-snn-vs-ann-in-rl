""" STDP Utility functions

"""
import sys
import string
from typing import Callable, Dict, Iterable, List, Tuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import snntorch as snn
from snntorch import surrogate
from snntorch import backprop
from snntorch import functional as SF
from snntorch import utils
from snntorch import spikeplot as splt
from snntorch import spikegen
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.functional as F

from stdp.spike_collectors import all_to_all
from stdp.tut_utils import *

sys.path.append("../")


def stdp_dW(
    A_plus: float,
    A_minus: float,
    tau_plus: float,
    tau_minus: float,
    delta_t: Iterable[int],
) -> float:
    """STDP Weight change formula

    $\deltat = t_pre - t_post$
    """
    dW = torch.zeros(len(delta_t))
    dW[delta_t <= 0] = A_plus * torch.exp(delta_t[delta_t <= 0] / tau_plus)
    dW[delta_t > 0] = -A_minus * torch.exp(-delta_t[delta_t > 0] / tau_minus)
    return dW


def stdp_generate_dw_lookup(dt_max: int):
    """
    """
    T_lu = torch.arange(-dt_max, dt_max+1)
    A_plus = 0.2
    tau_plus = 5e-3
    A_minus = 0.2
    tau_minus = 4.8e-3
    dw = stdp_dW(A_plus, A_minus, tau_plus, tau_minus, T_lu * 1e-5)
    T_lu = {int(dt): float(dwi) for dt, dwi in zip(T_lu.numpy().tolist(), dw)}
    T_lu[T_lu == 0] = 0
    return T_lu


def stdp_step_old(
    weights: torch.Tensor,
    connections: torch.Tensor,
    raster: torch.Tensor,
    dw_rule: str = "nearest_post_spike",
    bidirectional: bool = True,
    max_delta_t: int = 20,
    inplace: bool = False,
    v: bool = False,
) -> torch.Tensor:

    W = weights
    if not inplace:
        W = weights.clone()
    # if connections is not None: (Nearest Neighbors)
    # (All-to-all)
    if connections is None:
        connections = torch.full_like(W, 1)

    if bidirectional:
        # To consider bidirectional synapses just add the post-pre pairs and
        #   the rest does not change. The only thing expected is simply to
        #   see much smaller post-pre weights than pre-post weights
        pre_post = torch.argwhere(connections != 0).numpy().tolist()
    else:
        pre_post = torch.argwhere(connections > 0).numpy().tolist()
        """Pre-post pairs"""

    T_lu = stdp_generate_dw_lookup(max_delta_t)

    neurons_idx = torch.arange(W.shape[0])
    spks = [torch.argwhere(raster[i] == 1) for i in neurons_idx]
    """Spike positions for each neuron"""

    # ## Compute all the dw ##
    dts = {}
    dws = {}
    tpre_tpost = {}
    dts_all = {}  # stores all the `pre`-`post` spikes dt
    dws_all = {}  # stores all the `pre`-`post` spikes dw
    for pre, post in pre_post:  # neurons id
        if v:
            print(f"synapse: {pre} -> {post}")
        if post == pre:
            continue

        dts[pre, post] = 0
        dws[pre, post] = 0
        tpre_tpost[pre, post] = []
        dts_all[pre, post] = []
        dws_all[pre, post] = []
        post_spks = spks[post]
        if dw_rule in ["sum_all", "nearest_post_spike"]:

            for pre_si in spks[pre]:
                # each diffs between the i spk and each j spks
                diffs = pre_si - post_spks
                diffs = diffs[diffs != 0]  # filter out zeros
                diffs[diffs > max_delta_t] = max_delta_t  # clip - upper
                diffs[diffs < -max_delta_t] = -max_delta_t  # clip - lower

                #  Option 1: sum over all the spikes (Classical Additive)
                #    ref: eq.2 http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity#Basic_STDP_Model # noqa
                if dw_rule == "sum_all":
                    dt_ij = diffs
                    dw_ij = sum([T_lu[deltati.item()] for deltati in diffs])

                #  Option 2: take the closer spike (Nearest spike)
                elif dw_rule == "nearest_post_spike":
                    dt_ij = diffs[diffs.abs().argmin()].item()
                    dw_ij = T_lu[dt_ij]

                else:
                    raise ValueError(f"Unknown dw_rule: {dw_rule}")

                #  # apply additive rule
                dts[pre, post] += dt_ij
                dws[pre, post] += dw_ij

                # tpre_tpost[pre, post].append((pre_si, pre_si+dt_ij))
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

        elif dw_rule in ["nearest_pre_post_spikes"]:
            pass

    # ## Update the weights ##
    for pre_post, dw in dws.items():
        pre, post = pre_post
        W[pre, post] = W[pre, post] + dw

    return W


def stdp_step(
    weights: torch.Tensor,
    connections: torch.Tensor,
    raster: torch.Tensor,
    spike_collection_rule: callable = all_to_all,
    dw_rule: str = "sum",
    bidirectional: bool = True,
    max_delta_t: int = 20,
    inplace: bool = False,
    v: bool = False,
) -> torch.Tensor:
    """Simplified STDP step

    It assumes that a group of neurons is indexed by a unique integer.

    Parameters
    ----------
    weights : torch.Tensor
        2D Weights matrix. Each (i, j) position is the weight of the connection
        between the i-th neuron and the j-th neuron.
    connections : torch.Tensor
        2D Connections matrix. Each i-th row is a vector of
        connections with the other neurons. Each (i, j) position can be:
        * +1 if the i-th neuron is presynaptic and the j-th neuron is 
            postsynaptic;
        * -1 if the i-th neuron is postsynaptic and the j-th neuron is 
            presynaptic.
        * 0 if there is no connection between the i-th and the j-th neurons.
    raster : torch.Tensor
        2D Raster plot. Each i-th row is a vector of spike
        traces, j time-steps long.
    spike_collection_rule : callable, optional
        Default all_to_all
    dw_rule : str, optional
        By default "sum"
    bidirectional : bool, optional
        By default True
    max_delta_t : int, optional
        By default 20
    inplace : bool, optional
        By default False
    v : bool, optional
        If true the output will be more verbose. By default False

    Returns
    -------
    torch.Tensor
        _description_

    Raises
    ------
    ValueError
        _description_
    """    
    if raster.ndim != 2:
        raise ValueError(
            f"raster.ndim must be 2, instead it's shape is {raster.shape}")
    if raster.nelement() == 0:
        if v: print("stdp_step| Warning: the raster is empty")
        return weights
    W = weights
    if not inplace:
        W = weights.clone()
    # if connections is not None: (Nearest Neighbors)
    # (All-to-all)
    if connections is None:
        connections = torch.full_like(W, 1)

    if bidirectional:
        # To consider bidirectional synapses just add the post-pre pairs and
        #   the rest does not change. The only thing expected is simply to
        #   see much smaller post-pre weights than pre-post weights
        pre_post = torch.argwhere(connections != 0).numpy().tolist()
    else:
        pre_post = torch.argwhere(connections > 0).numpy().tolist()
        """Pre-post pairs"""

    T_lu = stdp_generate_dw_lookup(max_delta_t)

    # ## Compute all the dw ##
    hist = {
        'tpre_tpost': {},
        'dts_all': {},  # stores all the `pre`-`post` spikes dt
        'dws_all': {}  # stores all the `pre`-`post` spikes dw
    }
    for pre, post in pre_post:  # neurons id
        if v:
            print(f"synapse: {pre} -> {post}")
        if post == pre:
            continue

        pre_post_spks = spike_collection_rule(raster[pre, :], raster[post, :])

        # dt_ij = t_pre - t_post
        dt_ij = (-torch.tensor(pre_post_spks).diff()).flatten()
        dt_ij = torch.where(dt_ij > max_delta_t, max_delta_t, dt_ij)
        dt_ij = torch.where(dt_ij < -max_delta_t, -max_delta_t, dt_ij)
        dt_ij = dt_ij.tolist()
        dw_ij = [T_lu[dt_ij] for dt_ij in dt_ij]

        #  Option 1: sum over all the spikes (Classical Additive)
        #    ref: eq.2 http://www.scholarpedia.org/article/Spike-timing_dependent_plasticity#Basic_STDP_Model # noqa
        if dw_rule == "sum":
            dw_ij = sum(dw_ij)
        elif dw_rule == "prod":
            dw_ij = torch.prod(torch.tensor(dw_ij)).item()
        else:
            raise ValueError(f"Unknown dw_rule: {dw_rule}")

        hist['tpre_tpost'][pre, post] = pre_post_spks
        hist['dts_all'][pre, post] = dt_ij
        hist['dws_all'][pre, post] = dw_ij

        if v:
            print(f"{pre}, {post}")
            print(f"dws: {hist['dts_all'][pre, post]}")
            print(f"dws: {hist['dws_all'][pre, post]}")
            print("")
    # ## Update the weights ##
    for pre_post, dw in hist['dws_all'].items():
        pre, post = pre_post
        W[pre, post] = W[pre, post] + dw

    return W


def raster_collect_spikes(
    raster: torch.Tensor,
    collection_rule: Callable,
    connections: List[List[int]] or torch.Tensor = None
) -> Dict[tuple, list]:
    """Collect spikes from raster according to a collection rule

    Parameters
    ----------
    raster : torch.Tensor
        A raster matrix, where each row is a collection of spikes of a
        specific neuron, along time steps.
    collection_rule : Callable
        A function defining how to collect spikes between presynaptic neuron
        and postsynaptic one.
    connections : List[List[int]]ortorch.Tensor, optional
        Connections matrix. It defines the oriented connections from a 
        presynaptic neuron to a postsynaptic one, by a `1`.
        Defaults to None.

    Returns
    -------
    Dict[tuple, List(tuple)]
        {(pre, post): [(pre_spk_1, post_spk_1), ..., (pre_spk_i, post_spk_i)]}

    Raises
    ------
    ValueError
        If conncections is not square.
    ValueError
        If connections has not the same number of rows as raster.
    """
    raster = torch.Tensor(raster)
    if connections is None:
        connections = torch.ones(raster.shape[0], raster.shape[0])
    else:
        connections = torch.Tensor(connections)
    if connections.shape[0] != connections.shape[1]:
        raise ValueError("connections must be a square matrix")
    if connections.shape[0] != raster.shape[0]:
        raise ValueError(
            "connections and raster must have the same number of rows, got " +
            f"{connections.shape[0]} and {raster.shape[0]}"
        )

    pre_post = torch.argwhere(connections > 0).numpy().tolist()

    spks = {}
    for pre, post in pre_post:
        if post != pre:
            spks[pre, post]= collection_rule(raster[pre, :], raster[post, :])
    return spks





def model_get_named_layer_params(
        net, layer_name: str) -> torch.nn.parameter.Parameter:
    """Get layer weights by name of the layer

    Parameters
    ----------
    net : _type_
        The model
    layer_name : str
        The name of the layer in `net`.

    Returns
    -------
    torch.nn.parameter.Parameter
        A tensor with dimensions: (THIS_LAYER_NEURONS, PREV_LAYER_NEURONS)
    """
    return dict(net.named_parameters())[f'{layer_name}.weight']


def ridge_regression_get_coeffs(
        X: torch.TensorType, Y: torch.TensorType, alpha: float = 1):
    return torch.linalg.inv(
        X.T @ X + alpha * torch.eye(X.shape[1])) @ X.T @ Y

# Currently not used
def model_layers_to_weights(
        net, names: list) -> Dict[str, torch.nn.parameter.Parameter]:
    """
    CAVEAT: Only Sequential models at the moment
    """
    # Ws [=] { NAME: (PREV_LAYER_NEURONS, THIS_LAYER_NEURONS) }
    Ws ={ name: model_get_named_layer_params(net, name).T for name in names}
    # neurons_per_layer [=] {NAME: THIS_LAYER_NEURONS}
    neurons_per_layer = {name_i: w_ji.shape[1] for name_i, w_ji in Ws.items()}
    tot_neurons = sum(list(neurons_per_layer.values()))
    
    # layer_to_neurons_map [=] {NAME: (FROM_NEURON_IDX, TO_NEURON_IDX)}
    layer_to_neurons_map = {}
    ni = 0
    for name, nneurons in neurons_per_layer.items():
        layer_to_neurons_map[name] = (ni, ni + nneurons)
        ni += nneurons
    assert ni == tot_neurons, f"More neurons than present have been mapped"

    # M: total weights matrix
    M = torch.empty_like(tot_neurons, tot_neurons)


# Graphs #


def connections_to_digraph(
    connections: torch.Tensor,
    show: bool = True,
    ax: plt.Axes = None
) -> nx.MultiDiGraph:
    # TODO: fix position of the neurons
    G = nx.from_numpy_matrix(
        connections.numpy(),
        parallel_edges=True, create_using=nx.MultiDiGraph()
    )
    label_mapping = {
        i: string.ascii_lowercase[i] for i in range(connections.shape[0])}
    G = nx.relabel_nodes(G, label_mapping)
    G.edges(data=True)

    if show:
        if ax is None:
            _, ax = plt.subplots(1, 1)
        nx.draw(G, with_labels=True, ax=ax)
        return G, ax
    return G


# Visualization #

def get_equispaced_rgbas_from_cmap(
    n: int,
    cmap_name: str ='prism'
) -> List[tuple]:
    cmap = mpl.cm.get_cmap(cmap_name)
    return [cmap(i) for i in np.linspace(0, 1, n)]


def get_raster_from_spike_positions(
    spk_pos: Dict[int, Iterable[int]],
    size: Tuple[int, int]
) -> List[list]:
    """ Generate a raster from spike positions

    Example
    -------
    >>>spike_positions = {
    ...    0: [1, 3],
    ...    2: [2, 4]
    ...}
    >>>get_raster_from_spike_positions(spike_positions, size=(3, 5))
    [
        [0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 1]
    ]
    """
    spk_pos = dict(sorted(spk_pos.items(), key=lambda x: x[0]))
    if max(spk_pos.keys()) > size[0]:
        raise ValueError(
            f"Neuron number out of size: {max(spk_pos.keys())} > {size[0]}")

    raster = []
    for row in range(size[0]):
        if row in spk_pos.keys():
            if max(spk_pos[row]) > size[1]:
                raise ValueError(
                    "Spike position out of size: " +
                    f"{max(spk_pos[row])} > {size[1]}"
                )
            neuron_row = [
                1 if i in spk_pos[row] else 0 for i in range(size[1])]
            raster.append(neuron_row)
        else:
            raster.append([0] * size[1])
    return raster


def get_spike_positions_from_tpre_tpost(
    tpre_tpost: List[Iterable[int]],
    neurons: List[int]
) -> Dict[int, List[int]]:
    """Get spike positions from tpre_tpost

    Example
    -------
    >>>tpre_tpost = [
    ...    (0, 2),
    ...    (2, 2),
    ...    (4, 5)
    ...]
    >>>neurons = [4, 6]
    >>>get_spike_positions_from_tpre_tpost(tpre_tpost, neurons)
    {
        4: [0, 2, 4],
        6: [2, 5]
    }
    """
    if np.ndim(tpre_tpost) != 2:
        raise ValueError("tpre_tpost must be a List of Iterables")
    lengths = set([len(x) for x in tpre_tpost])
    if len(lengths) != 1:
        raise ValueError("tpre_tpost must contain tuples of the same size")
    if len(neurons) > len(tpre_tpost[0]):
        raise ValueError(
            f"Neurons number out of size: {max(neurons)} > {len(tpre_tpost)}")

    return {
        n: list(set([x[i] for x in tpre_tpost])) for i, n in enumerate(neurons)
    }


def get_raster_from_tpre_tpost(
    tpre_tpost: List[Iterable[int]],
    neurons: List[int],
    size: Tuple[int, int] = None
) -> List[list]:
    """ Generate a raster from tpre_tpost

    Example
    -------
    >>>tpre_tpost = [
    ...    (0, 2),
    ...    (2, 2),
    ...    (4, 5)
    ...]
    >>>neurons = [4, 6]
    >>>get_raster_from_tpre_tpost(tpre_tpost, neurons)
    [
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0],
        [1, 0, 1, 0, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 1]
    ]
    """
    if size is None:
        size = (
            max(neurons) + 1,
            max([max(x) for x in tpre_tpost]) + 1
        )
    if np.ndim(tpre_tpost) != 2:
        raise ValueError("tpre_tpost must be a List of Iterables")

    spk_pos = get_spike_positions_from_tpre_tpost(tpre_tpost, neurons)
    return get_raster_from_spike_positions(
        spk_pos,
        size=size
    )





def plot_raster(
    raster: np.array,
    dt: None or float = None,
    colors: Iterable = None,
    linelengths: Iterable = None,
    title: str = "Raster plot",
    ylabel: str = "Neuron",
    xlabel: str = None,
    show_horizontal_lines: bool = True,
    ax: plt.Axes = None,
) -> Tuple[List[mpl.collections.EventCollection], plt.Axes]:
    """Plot a raster plot

    Args:
        raster (np.array): A numpy array N x T, where each row represent a
            distinct neuron time sequence of spikes
        dt (Noneorfloat): The time delta of each time step, in seconds.
            colors (Iterable, optional): A collection of colors, where each
            element represent the color of a specific neuron raster plot. Defaults
            to None.
        linelengths (Iterable, optional): A collection of line lengths,
            where each element represents the line length of the spikes of a
            specific neuron raster plot. Defaults to None.
            title (str, optional): The title of the plot. Defaults to
            'Raster plot'.
            ylabel (str, optional): Y label. Defaults to 'Neuron'.
            xlabel (str, optional): X label. Defaults to None.
            show_horizontal_lines (bool, optional): If True show boundary lines
            between neurons raster plots. Defaults to True.

    Returns:
        Tuple[List[mpl.collections.EventCollection], plt.Axes]:
    """
    spike_pos = [np.argwhere(row).flatten() for row in raster][::-1]
    if isinstance(dt, float):
        spike_pos = [spi * dt for spi in spike_pos]

    # Styles #
    if colors is None:
        colors = [(0, 0, 0) for i in range(raster.shape[0])]
    if linelengths is None:
        linelengths = [0.5 for i in range(raster.shape[0])]

    # Plot #
    if ax is None:
        fig, ax = plt.subplots(1, 1)

    raster_rows = ax.eventplot(
        spike_pos, color=colors, linelengths=linelengths)

    # Horizontal Lines #
    if show_horizontal_lines:
        for ni in range(raster.shape[0]):
            ax.axhline(ni + 0.65, color="gray", linewidth="0.5")

    # Labels #
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.yaxis.set_major_locator(mpl.ticker.FixedLocator(range(raster.shape[0])))
    if isinstance(dt, float):
        ax.set_xlabel("t")
        # ax.xaxis.set_major_locator(
        #   mpl.ticker.FixedLocator(np.arange(raster.shape[1]))*dt)
        # ax.set_xticklabels([ti * dt for ti in range(raster.shape[1])])
    else:
        ax.set_xlabel("$t_{step}$")
        ax.xaxis.set_major_locator(
            mpl.ticker.FixedLocator(range(raster.shape[1])))
    if xlabel is not None and type(xlabel) == str:
        ax.set_xlabel(xlabel)
    return raster_rows, ax


def plot_composed_rasters_from_tpre_tpost_groups(
    tpre_tpost_groups: Dict[tuple, List[Iterable]],
    groups_colors: List[tuple] = None,
    size: List[int] = None,  # raster shape
    lines_alpha = 0.9,  # set transparency to show superimposed ones
    linelengths: Iterable = None,
    title: str = "Raster plot",
    ylabel: str = "Neuron",
    xlabel: str = None,
    show_horizontal_lines: bool = True,
    ax: plt.Axes = None
) -> plt.Axes:
    # TODO: adapt interface to plot_raster
    if type(tpre_tpost_groups) != dict:
        raise ValueError(f"`tpre_tpost_groups` must be a dict")

    if groups_colors is None:
        groups_colors = get_equispaced_rgbas_from_cmap(n=len(tpre_tpost_groups))

    #TODO: if size is None: ...

    if linelengths is None:
        linelengths = [0.5 for _ in range(size[0])]

    if ax is None:
        _, ax = plt.subplots(1, 1)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if xlabel is not None and type(xlabel) == str:
            ax.set_xlabel(xlabel)

    # Horizontal Lines #
    if show_horizontal_lines:
        for ni in range(size[0]):
            ax.axhline(ni + 0.65, color="gray", linewidth="0.5")
    
    for i, neurons in enumerate(tpre_tpost_groups):
        tpre_tpost = tpre_tpost_groups[neurons]
        partial_raster = get_raster_from_tpre_tpost(tpre_tpost, neurons, size=list(size))
        spike_pos = [np.argwhere(row).flatten() for row in partial_raster][::-1]
        color = [*groups_colors[i][:3], lines_alpha]
        ax.eventplot(
            spike_pos,
            color=[color] * len(partial_raster),
            linelengths=linelengths
        )
    return ax