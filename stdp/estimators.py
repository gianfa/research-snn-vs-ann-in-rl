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
            input_scaling: float = 1.,
            # hidden_sparsity: float= 0,
            spectral_radius=0.95) -> None:
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
        self.input_scaling = input_scaling

        # TODO:
        #     maybe admit a W_in as a parameter,
        #     in order to selectively deactivate neurons
        self.W_in = nn.Linear(input_size, hidden_size, bias=False)
        self.W = nn.Linear(hidden_size, hidden_size, bias=False)
        self.W_out = nn.Linear(hidden_size, output_size, bias=False)
        for param in self.W_in.parameters():
            param.requires_grad = False
        for param in self.W.parameters():
            param.requires_grad = False

        self.hist = {}  # Last training history object

        self.init_weights(hidden_weights)

    def init_weights(
            self,
            hidden_weights: torch.TensorType or str = "orthogonal") -> None:
        # TODO: replace with https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.orthogonal_
        nn.init.orthogonal_(self.W_in.weight)
        if hidden_weights is None or hidden_weights == "orthogonal":
            nn.init.orthogonal_(self.W.weight)
        else:
            self.W.weight.data = hidden_weights
        self.W.weight = nn.Parameter(
            self.W.weight * (
                self.spectral_radius / 
                    torch.max(
                        torch.abs(
                            torch.linalg.eig(self.W.weight).eigenvalues))))
        nn.init.normal_(self.W_out.weight, std=0.01)
        return None

    def forward(self,
            input: torch.Tensor,
            activation: Callable = torch.tanh,
        ) -> torch.Tensor:
        input *= self.input_scaling

        # Compute Dynamics: Iteratively pass through the input and hidden
        time_length = input.shape[0]
        x = torch.zeros(self.hidden_size)
        x_0 = x.clone() # storing for diagnosis

        for t in range(time_length):
            u = input[t]  # current input
            x = activation(self.W_in(u) + self.W(x))  # internal state

        if bool(input.sum() != 0):
            assert not x.equal(x_0), (
                f"The dynamics was skipped (no weights changes). " +
                "Check the code")
        # Last activations go to the output
        output = self.W_out(x).unsqueeze(1)
        return output


    def train(
            self,
            X: torch.TensorType,
            Y: torch.TensorType,
            epochs: int = 10,
            optimizer: Callable = torch.optim.SGD,
            lr: float = 4e-2,
            criterion: Callable = torch.nn.MSELoss(),
            activation: Callable = torch.tanh,
            v: bool = False) -> None:
        # TODO: add hooks for weights storage
        optimizer = optimizer(self.W_out.parameters(), lr=lr)
        self.hist['losses'] = []
        for epoch in range(epochs):
            for X_i, Y_i in zip(X, Y):
                X_i, Y_i = X_i.to(torch.float32), Y_i.to(torch.float32)
                if X_i.ndim == 1: X_i = X_i.unsqueeze(0)  # expand as a row
                if Y_i.ndim == 1: Y_i = Y_i.unsqueeze(0)  # expand as a row

                out = self.__call__(X_i, activation=activation)
                loss = criterion(out, Y_i)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                self.hist['losses'].append(loss)
            if v:
                print(
                    f"Weights after epoch {epoch}: " +
                    f"{self.W_out.weight.data.mean()}")

        if v:
            try:
                fig, ax = plt.subplots()
                ax.plot(
                    torch.stack(self.hist['losses']).detach().numpy())
                title = "Loss"
                if isinstance(optimizer, torch.optim.Optimizer):
                    title = f"{optimizer.__class__.__name__} {title}"
                ax.set_title(f"{title}")
                ax.set_xlabel("# iteration")
                
            except NameError as e:
                print("WARN| Matplotlib not loaded or missing")


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