""" STDP Utility functions

"""
import sys
import string
from typing import Iterable, List, Tuple
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
    T_lu = torch.arange(-dt_max, dt_max)
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
        dt_ij = (-torch.tensor(pre_post_spks).diff()).flatten().tolist()
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


# Graphs #


def connections_to_digraph(
    connections: torch.Tensor, show: bool = True
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
        nx.draw(G, with_labels=True)
    return G


# Visualization #


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
