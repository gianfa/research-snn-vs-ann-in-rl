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

from stdp.spike_collectors import all_to_all, nearest_pre_post_pair
from stdp.tut_utils import *

sys.path.append("../")


def generate_random_connections_mask(
        shape: tuple, density: float) -> torch.Tensor:
    mask = torch.rand(shape[0], shape[1])
    mask[mask > density] = 0
    return mask


def generate_simple_circle_connections_mask(shape: tuple) -> torch.Tensor:
    mask = torch.diag(torch.rand(shape[0] - 1), diagonal=-1)
    mask[0, -1] = torch.rand(1)
    return mask



# STDP

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


def stdp_generate_dw_lookup(
        dt_max: int, time_related=True):
    """
    """
    if time_related:
        T_lu = torch.arange(-dt_max, dt_max+1)
        A_plus = 0.2
        tau_plus = 5e-3
        A_minus = 0.2
        tau_minus = 4.8e-3
        dw = stdp_dW(A_plus, A_minus, tau_plus, tau_minus, T_lu * 1e-5)
        T_lu = {int(dt): float(dwi) for dt, dwi in zip(T_lu.numpy().tolist(), dw)}
        T_lu[T_lu == 0] = 0
    else:
        CONSTANT = 1.2
        T_lu = torch.zeros(dt_max * 2 + 1)
        T_lu[T_lu == 0] = CONSTANT #
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
    time_related: bool = True,
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

    W = weights
    if not inplace:
        W = weights.clone()
    # apply connections mask
    W *=  (connections != 0).int()

    T_lu = stdp_generate_dw_lookup(max_delta_t, time_related=time_related)
    print(f"nonzero: {(W != 0).int().sum()}")
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
    relabel = False,
    ax: plt.Axes = None
) -> nx.MultiDiGraph:
    # TODO: fix position of the neurons
    G = nx.from_numpy_matrix(
        connections.numpy(),
        parallel_edges=True, create_using=nx.MultiDiGraph()
    )

    if relabel:
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


def plot_most_changing_node_weights(
    W_hist: torch.Tensor,
    n_top_weights: int = 5,
    axs: List[plt.Axes] = None
):
    if type(W_hist) == torch.Tensor:
        W_hist = W_hist.detach().numpy()

    if axs is None:
        _, axs = plt.subplots(2, 1)
    fig = axs[0].figure

    abs_diff = np.abs( np.diff(W_hist, axis=0) ).sum(axis=0)
    if abs_diff.sum() == 0:
        print("INFO: No difference between W along hist")
    else:
        flat_indices = np.argsort(abs_diff, axis=None)[-n_top_weights:]
        row_indices, col_indices = np.unravel_index(
            flat_indices, abs_diff.shape)

    for ri, ci in zip(row_indices, col_indices):
        axs[0].plot(W_hist[:, ri, ci], "-o", label=f"{ri} -> {ci}")
    axs[0].grid()
    axs[0].legend()
    axs[0].set(
        title=f'Top {n_top_weights} changing weights',
        xlabel=f"# epoch"
    )

    # Plot Topology
    nodes = set(col_indices.tolist() + row_indices.tolist()); print(nodes)
    edges = [(row, col) for row, col in zip(row_indices, col_indices)]
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_edges_from(edges)
    pos = nx.spring_layout(G)  # Posizionamento dei nodi nel layout

    nx.draw_networkx(G, pos, with_labels=True, arrows=True, ax=axs[1])
    axs[1].set(title='Topology of the most changing weights')
    
    fig.tight_layout()
    return axs


#


def single_index_to_coordinate(index, num_columns):
    row = index // num_columns
    col = index % num_columns
    return row, col


def generate_sparse_matrix(shape, density, values_f=torch.rand):
    num_elements = int(shape[0] * shape[1] * density)
    indices = torch.randperm(shape[0] * shape[1])[:num_elements]
    indices = torch.Tensor([
        single_index_to_coordinate(i, shape[1]) for i in indices])
    values = values_f(num_elements)
    matrix = torch.sparse_coo_tensor(indices.T, values, torch.Size(shape))
    return matrix

#


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


# STDP Helpers #


def get_named_layer_parameters(model: nn.Module, layer_name: str) -> dict:
    return {l_name: l_par for l_name, l_par in model.named_parameters()
            if l_name == layer_name}


def get_named_layer(model: nn.Module, layer_name: str) -> dict:
    return [l[1] for l in model.named_modules() if l[0] == layer_name][0]


def register_activations_hooks_to_layers(
        model: nn.Module,
        steps_to_reset: int or None = None, v: bool = False) -> \
    List[torch.utils.hooks.RemovableHandle]:

    def init_activations() -> dict:
        return {
            name: [] for name, _ in model.named_modules() if len(name) > 0}

    def get_activations(layer, input, output, label: str):
        if steps_to_reset and len(activations[label]) >= steps_to_reset:
            activations[label].clear()
        activations[label].append(output)

    activations = init_activations()

    hooks = []
    for name, layer in model.named_modules():
        if len(name) > 0:
            hooks.append(
                layer.register_forward_hook(
                partial(get_activations, label=name)))
            if v: print(f"Registered hook to layer: {name}")
    return activations, hooks


def activations_as_tensors(activations: dict):
    return {layer_name: torch.vstack(acts)
                for layer_name, acts in activations.items()}


def traces_ensure_are_3_dim(traces: dict) -> dict:
    return {
        name:(trace if trace.ndim==3 else trace.unsqueeze(2))
        for name, trace in traces.items()}


def traces_contract_to_2_dim(traces: dict) -> dict:
    """{LAYER: tensor(nbatches x nsteps, nunits)}"""
    return {
        name: trace.view(-1, trace.size(2))
        for name, trace in traces_ensure_are_3_dim(traces).items()}


def layertraces_to_traces(traces: dict) -> dict:
    return torch.vstack(
        [trace.T for trace in traces_contract_to_2_dim(traces).values()])


def activations_to_traces(activations: dict, th: float) -> torch.Tensor:
    acts_tens = activations_as_layertraces(activations, th)
    return layertraces_to_traces(acts_tens)


def activations_as_layertraces(activations: dict, th: float) -> dict:
    return {layer_name: torch.vstack(acts) > th
                for layer_name, acts in activations.items()}


def activations_to_traces(activations: dict, th: float) -> torch.Tensor:
    return  torch.hstack( list(
                    activations_as_layertraces(activations, th).values()))


def clear_activations(activations: dict) -> dict:
    return {name: [] for name, acts in activations.items()}
