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

from stdp.funx import *
from stdp.spike_collectors import all_to_all
from stdp.tut_utils import *


class ESN(nn.Module):
    def __init__(
            self,
            input_size: int,
            hidden_size: int,
            output_size: int,
            hidden_weights: torch.TensorType = None,
            spectral_radius=0.9) -> None:
        super(ESN, self).__init__()
        if (
            hidden_weights is not None and 
                (
             hidden_weights.size() != torch.Size((hidden_size, hidden_size)))):
            raise ValueError(
                "hidden weights shape must be input_size x hidden_size")
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.spectral_radius = spectral_radius

        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out = nn.Linear(hidden_size, output_size, bias=False)

        self.init_weights(hidden_weights)

    def init_weights(self, hidden_weights: torch.TensorType = None) -> None:
        nn.init.orthogonal_(self.W_in.weight)
        if hidden_weights is None:
            nn.init.orthogonal_(self.W.weight)
        else:
            self.W.weight.data = hidden_weights
        self.W.weight = nn.Parameter(
            self.W.weight * (
                self.spectral_radius / 
                    torch.max(
                        torch.abs(torch.eig(self.W.weight).eigenvalues))))
        nn.init.normal_(self.W_out.weight, std=0.01)
        return None

    def forward(self,
            input: torch.Tensor,
            activation: Callable = torch.tanh) -> torch.Tensor:
        time_length = input.shape[0]
        x = torch.zeros(self.hidden_size)

        for t in range(time_length):
            u = input[t]  # current input
            x = activation(self.W_in(u) + self.W(x))  # internal state
        output = self.W_out(x)
        return output

def define_spiking_cluster(
    tot_neurons: int = 100,
    input_neuron_idxs: List[int] = None,
    output_neuron_idxs: List[int] = None,
    weights: torch.Tensor = None
) -> dict:
    """
    Assumptions:
        weights in R
        pre->post (ij, ji)
        exc/inh   (<0 inh; >0 exc)
    """
    if (input_neuron_idxs is not None and
        output_neuron_idxs is not None and
        input_neuron_idxs + output_neuron_idxs > tot_neurons):
            raise ValueError(
                "input_neuron_idxs + output_neuron_idxs must be <= tot_neurons")
    if (input_neuron_idxs is not None and
        input_neuron_idxs > tot_neurons):
            raise ValueError(
                "input_neuron_idxs must be <= tot_neurons")
    if (output_neuron_idxs is not None and
        output_neuron_idxs > tot_neurons):
            raise ValueError(
                "output_neuron_idxs must be <= tot_neurons")
    
    in_id = input_neuron_idxs
    out_id = output_neuron_idxs
    if in_id is None or out_id is None:
        random_idxs = np.random.permutation(range(tot_neurons))
        max_in_idx = int(np.floor(tot_neurons * 0.6))
        max_out_idx = int(np.floor(tot_neurons * 0.4))
        # TODO: TOTEST
        if out_id is not None:
            random_idxs = random_idxs[len(out_id):]
        if in_id is None:
            in_id = list(random_idxs[:np.random.randint(0, max_in_idx, 1)[0]])
        
        random_idxs = random_idxs[len(in_id):]
        if out_id is None:
            out_id = list(
                random_idxs[:np.random.randint(0, max_out_idx, 1)[0]])
        
    if weights is None:
        SPARSITY = 0.2
        weights = torch.rand(tot_neurons, tot_neurons)
        #TODO: weights[random 0.2 coords] = 0

    return {
        "weights": weights,
        "input": in_id,
        "output": out_id,
    }


class Reservoir(nn.Module):
    """
    This class is designed to create a reservoir.
    It implements an input layer and an output layer, which connect the
    relative of neurons (input and ouput) with the outside world.
    These two layers are logical layers, that is, the neurons in these
    layers are an integral part of the bulk of neurons, but for descriptive
    convenience we describe them externally.
    """
    def __init__(
            self,
            input_size: int = 10,  # input example size
            tot_neurons: int = 100,
            input_neuron_idxs: List[int] = None,
            output_neuron_idxs: List[int] = None,
            weights: torch.Tensor = None,
            beta: torch.Tensor = None
        ):
        super().__init__()

        clust = define_spiking_cluster(
            tot_neurons,
            input_neuron_idxs,
            output_neuron_idxs,
            weights,
        )

        weights = clust['weights']
        in_ns = clust['input']
        out_ns = clust['output']
        del clust

        assert weights.ndim == 2
        assert weights.shape[0] == weights.shape[1]

        self.in_ns = in_ns
        self.out_ns = out_ns
        self.weights = weights
        beta = beta or 0.95

        # Initialize layers
        self.input = nn.Linear(input_size, weights.shape[0])
        self.input.name = 'input'
        self.input_lif = snn.Leaky(beta=beta)
        self.input_lif.name = 'input-lif'

        self.cluster = nn.Linear(weights.shape[0], weights.shape[0])
        self.cluster.name = 'cluster'
        self.cluster_lif = snn.Leaky(beta=beta)
        self.cluster_lif.name = 'cluster-lif'

        # previous states storage
        self.prev = {'spk': None, 'mem': None}

    def __len__(self):
        return self.weights.shape[0]

    def forward(self, x):
        """


        """
        in_cur1 = self.input(x)
        spk1, mem1 = self.input_lif(in_cur1, mem1)

        clust_cur = self.cluster(spk1)
        spk2, mem2 = self.lif2(clust_cur, mem2)

        # TODO: 0. input = concat( self.prev['mem'] - x, x )
        #           where x is involved only on the in_ns
        # -----
        # cur1 = self.fc1(x)
        # spk1, mem1 = self.lif1(cur1, mem1)
        # -----
        # TODO: 1. spk, mem = LIF(input)
        # done
        return